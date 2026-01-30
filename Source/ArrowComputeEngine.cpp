#include <BOSS.hpp>
#include <Expression.hpp>
#include <ExpressionUtilities.hpp>
#include <Utilities.hpp>
#include <algorithm>
#include <arrow/api.h>

#include <arrow/acero/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <memory>
#include <random>
#include <unordered_map>
#include <set>
using namespace boss::utilities::experimental;
using namespace arrow;
using namespace acero;
using arrow::Table;
using arrow::acero::Declaration;
using boss::Symbol;
using boss::utilities::operator""_;
using boss::ComplexExpression;
using boss::Expression;

class ColumnConverter {
  boss::ExpressionArguments columnValues;

  template <typename T, typename = void> struct has_value : std::false_type {};
  template <typename T>
  struct has_value<T, std::void_t<decltype(std::declval<T>().value)>> : std::true_type {};

public:
  template <typename T> arrow::Status Visit(T& value) {
    if constexpr(has_value<T>::value) {
      if constexpr(std::is_convertible_v<decltype(value.value),
                                         boss::expressions::AtomicExpression>) {
        columnValues.push_back(value.value);
      } else if constexpr(std::is_same_v<std::remove_reference_t<T>, arrow::StringScalar const>) {
        columnValues.push_back(
            (value.data() ? boss::Expression((char const*)value.data()) : "null"_));
      } else {
        columnValues.push_back("ArrowType"_(value.ToString()));
      };
    };
    return arrow::Status::OK();
  }
  boss::ExpressionArguments getColumnValues() && { return std::move(columnValues); };
};

static class {
  std::unordered_map<size_t, Declaration> intermediates;
  std::unordered_map<boss::Symbol, size_t> names;

  int64_t generateID() {
    static std::default_random_engine generator;
    return std::uniform_int_distribution<int64_t>(LONG_LONG_MIN, LONG_LONG_MAX)(generator);
  };

public:
  boss::Expression name(boss::Expression&& key, boss::Symbol name) {
    names[name] = std::get<int64_t>(key);
    return std::move(key);
  }
  int64_t byName(boss::Symbol name) { return names.at(name); }

  int64_t put(Declaration table) {
    auto id = generateID();
    intermediates[id] = table;
    return id;
  };
  Declaration at(boss::Expression const& key) { return intermediates.at(std::get<int64_t>(key)); }

  boss::Expression convertResult(boss::Expression const& key) {
    if(!std::holds_alternative<int64_t>(key))
      return key.clone(boss::expressions::CloneReason::EVALUATE_CONST_EXPRESSION);
    auto table = *arrow::acero::DeclarationToTable(at(key), false);
    boss::ExpressionArguments resultExpression;
    for(auto i = 0u; i < table->num_columns(); i++) {
      auto resultColumn = table->column(i);
      auto visitor = ColumnConverter();
      for(auto i = 0u; i < resultColumn->length(); i++) {
        auto _ = arrow::VisitScalarInline(*resultColumn->GetScalar(i).ValueUnsafe(), &visitor);
      }
      resultExpression.push_back(ComplexExpression(boss::Symbol(table->field(i)->name()),
                                                   std::move(visitor).getColumnValues()));
    }
    return ComplexExpression("Table"_, {}, std::move(resultExpression));
  };
  std::shared_ptr<arrow::Table> getTable(Declaration d) {
    return d.factory_name == "table_source"
               ? dynamic_cast<arrow::acero::TableSourceNodeOptions*>(d.options.get())->table
               : (*DeclarationToTable(d, false));
  };

  void collectGarbage() {
    auto keysToDelete = std::set<size_t>();
    for(auto& [key, value] : intermediates)
      keysToDelete.insert(key);
    for(auto& [name, key] : names)
      keysToDelete.erase(key);
    for(auto& key : keysToDelete)
      intermediates.erase(key);
  }
} intermediates;

static boss::Expression evaluate(boss::Expression&& e) {
  static auto _ = compute::Initialize();
  return std::move(e) //
      <"Slice" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
        return intermediates.put(Declaration::Sequence(
            {intermediates.at(dynamics.at(0)),
             {"fetch", FetchNodeOptions(get<int>(dynamics.at(1)), get<int>(dynamics.at(2)))}}));
      } //
  <"OrderBy" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    auto orderKeys = std::vector<compute::SortKey>();
    for(auto& it : get<ComplexExpression>(dynamics.at(1)).getDynamicArguments())
      orderKeys.push_back(compute::SortKey(get<Symbol>(it).getName()));
    return intermediates.put(Declaration::Sequence(
        {intermediates.at(dynamics.at(0)), {"order_by", OrderByNodeOptions(Ordering(orderKeys))}}));
  } //
  <"Join" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    auto leftKeys = std::vector<FieldRef>(), rightKeys = std::vector<FieldRef>();
    for(auto& it : get<ComplexExpression>(dynamics.at(1)).getDynamicArguments())
      leftKeys.push_back(get<Symbol>(it).getName());
    for(auto& it : get<ComplexExpression>(dynamics.at(3)).getDynamicArguments())
      rightKeys.push_back(get<Symbol>(it).getName());
    return intermediates.put({"hashjoin",
                              {intermediates.at(dynamics.at(0)), intermediates.at(dynamics.at(2))},
                              HashJoinNodeOptions(JoinType::INNER, leftKeys, rightKeys,
                                                  literal(true), "_l", "_r", true)});
  } //
  <"Name" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    return intermediates.name(std::move(dynamics.at(0)), get<boss::Symbol>(dynamics.at(1)));
  } //
  <"ByName" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    return intermediates.byName(get<boss::Symbol>(dynamics.at(0)));
  } //
  <"Project" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    auto projections = std::vector<compute::Expression>();
    auto names = std::vector<std::string>();
    for(auto i = 1; i < dynamics.size(); i++)
      visit(boss::utilities::overload(
                [&](Symbol&& s) {
                  projections.push_back(compute::field_ref(s.getName()));
                  names.push_back(s.getName());
                },
                [&](ComplexExpression&& s) {
                  auto arguments = std::vector<compute::Expression>();
                  for(auto& it :
                      s.getDynamicArguments()) { // TODO: implement deeply nested expressions
                    visit(boss::utilities::overload(
                              [&arguments](Symbol const&& s) {
                                arguments.push_back(compute::field_ref(s.getName()));
                              },
                              [&arguments](int const&& s) {
                                arguments.push_back(compute::literal(s));
                              },
                              [](auto&& s) { __builtin_trap(); }),
                          std::move(it));
                  };
                  projections.push_back(
                      s.getHead().getName() == "int"
                          ? compute::call("cast", arguments, compute::CastOptions::Unsafe(int32()))
                          : compute::call(s.getHead().getName(), arguments));
                  names.push_back(s.getHead().getName() == "int"
                                      ? "int(" + arguments.back().ToString() + ")"
                                      : compute::call(s.getHead().getName(), arguments).ToString());
                },
                [](auto&&) {}),
            std::move(dynamics.at(i)));
    return intermediates.put(Declaration::Sequence(
        {intermediates.at(dynamics.at(0)), {"project", ProjectNodeOptions(projections, names)}}));
  } //
  <"GroupBy" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    auto const& aggregationFunction = get<ComplexExpression>(dynamics.at(1));
    auto const aggregation = get<Symbol>(aggregationFunction.getDynamicArguments().at(0));
    return intermediates.put(Declaration::Sequence(
        {intermediates.at(dynamics.at(0)),
         {"aggregate",
          AggregateNodeOptions(
              {compute::Aggregate(
                  (dynamics.size() == 3 ? "hash_" : "") + aggregationFunction.getHead().getName(),
                  {aggregation.getName()},
                  aggregationFunction.getHead().getName() + "_" + aggregation.getName())},
              dynamics.size() == 3 ? std::vector<FieldRef>{(get<Symbol>(dynamics.at(2)).getName())}
                                   : std::vector<FieldRef>())}}));
  } //
  <"Cumulate" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    auto const& aggregationFunction = get<ComplexExpression>(dynamics.at(1));
    auto const aggregationAttribute = get<Symbol>(aggregationFunction.getDynamicArguments().at(0));
    auto input = (intermediates.getTable(intermediates.at(dynamics.at(0))));
    auto result = compute::CallFunction("cumulative_" + aggregationFunction.getHead().getName(),
                                        {input->GetColumnByName(aggregationAttribute.getName())})
                      ->chunked_array();
    return intermediates.put({"table_source", TableSourceNodeOptions(*input->AddColumn(
                                                  input->num_columns(),
                                                  field(aggregationFunction.getHead().getName() +
                                                            "_" + aggregationAttribute.getName(),
                                                        result->type()),
                                                  result))});
  } //
  <"Pairwise" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    auto options = compute::PairwiseOptions(get<int>(dynamics.at(3)));
    auto input = intermediates.getTable(intermediates.at(dynamics.at(0)));
    auto column = get<Symbol>(dynamics.at(2)).getName();
    auto inputArray =
        (input->GetColumnByName(column)->num_chunks() == 1 ? input : *input->CombineChunks())
            ->GetColumnByName(column);
    auto result = compute::CallFunction("pairwise_diff", {inputArray->chunk(0)}, &options);
    return intermediates.put(
        {"table_source",
         TableSourceNodeOptions(*input->AddColumn(
             input->num_columns(), field(get<Symbol>(dynamics.at(1)).getName(), result->type()),
             *ChunkedArray::Make({result->make_array()})))});
  } //
  <"ToStatus" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    return DeclarationToStatus(intermediates.at(dynamics.at(0)), false).CodeAsString();
  } //
  <"Materialize" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    return intermediates.put(
        {"table_source",
         TableSourceNodeOptions(
             *intermediates.getTable(intermediates.at(dynamics.at(0)))->CombineChunks())});
  } //
  <"Load" >= Recurse(evaluate)>[](auto, auto dynamics, auto) {
    return std::visit(
        boss::utilities::overload(
            [](std::string&& path) -> boss::Expression {
              auto options = csv::ReadOptions::Defaults();
              options.use_threads = false;
              options.block_size = 1 << 26;
              auto maybeTable =
                  (*csv::TableReader::Make(io::default_io_context(), *io::ReadableFile::Open(path),
                                           options, csv::ParseOptions::Defaults(),
                                           csv::ConvertOptions::Defaults()))
                      ->Read();
              if(!maybeTable.ok())
                return maybeTable.status().ToStringWithoutContextLines();
              return intermediates.put(
                  {"table_source", TableSourceNodeOptions(*(*maybeTable)->CombineChunks())});
            },
            [](auto&& e) -> boss::Expression { return e; }),
        std::move(dynamics.at(0)));
  };
};

extern "C" BOSSExpression* evaluate(BOSSExpression* e) {
  auto result =
      new BOSSExpression{.delegate = intermediates.convertResult(evaluate(std::move(e->delegate)))};
  intermediates.collectGarbage();
  return result;
};
