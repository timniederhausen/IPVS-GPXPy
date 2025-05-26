#pragma once

#include <boost/json.hpp>
#include <vector>

// Struct containing all results we'd like to compare
struct gprat_results
{
    std::vector<std::vector<double>> choleksy;
    std::vector<double> losses;
    std::vector<std::vector<double>> sum;
    std::vector<std::vector<double>> full;
    std::vector<double> pred;
};

// The following two functions are for JSON (de-)serialization
inline void tag_invoke(boost::json::value_from_tag, boost::json::value &jv, const gprat_results &results)
{
    jv = {
        { "choleksy", boost::json::value_from(results.choleksy) },
        { "losses", boost::json::value_from(results.losses) },
        { "sum", boost::json::value_from(results.sum) },
        { "full", boost::json::value_from(results.full) },
        { "pred", boost::json::value_from(results.pred) },
    };
}

// This helper function deduces the type and assigns the value with the matching key
template <typename T>
BOOST_FORCEINLINE void extract(const boost::json::object &obj, T &t, std::string_view key)
{
    t = boost::json::value_to<T>(obj.at(key));
}

inline gprat_results tag_invoke(boost::json::value_to_tag<gprat_results>, const boost::json::value &jv)
{
    gprat_results results;
    const auto &obj = jv.as_object();
    extract(obj, results.choleksy, "choleksy");
    extract(obj, results.losses, "losses");
    extract(obj, results.sum, "sum");
    extract(obj, results.full, "full");
    extract(obj, results.pred, "pred");
    return results;
}
