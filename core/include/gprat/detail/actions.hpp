﻿#ifndef GPRAT_DETAIL_ACTIONS_HPP
#define GPRAT_DETAIL_ACTIONS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <hpx/async_distributed/dataflow.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>

GPRAT_NS_BEGIN

/// @brief This template provides access to a function F's associated HPX action and related metadata.
///
/// Users can use this template to access the previously declared HPX plain (and optionally direct) action.
/// This way we get singleton-like semantics for free, there is always only one plain action associated with
/// a Callable value F.
template <auto F>
struct plain_action_for;

#define GPRAT_DECLARE_PLAIN_ACTION_FOR(local_function, action, friendly_name)                                          \
    template <>                                                                                                        \
    struct plain_action_for<local_function>                                                                            \
    {                                                                                                                  \
        using action_type = action;                                                                                    \
        constexpr static std::string_view name = friendly_name;                                                        \
    }

#define GPRAT_DEFINE_PLAIN_ACTION_FOR(local_function)

// =============================================================
// distributed action-based scheduling

struct tiled_scheduler_distributed
{
    /// @brief Create a new scheduler that targets all localities.
    tiled_scheduler_distributed() :
        localities_(hpx::find_all_localities())
    {
        // ctor
    }

    /// @brief Create a new scheduler that targets the given localities.
    explicit tiled_scheduler_distributed(std::vector<hpx::id_type> in_localities) :
        localities_(std::move(in_localities))
    {
        // ctor
    }

    std::vector<hpx::id_type> localities_;
};

namespace detail
{
// HPX does not auto-collapse future chains in their async(), dataflow(), ... functions.
// This usually works fine, but we require shared_future<R>s most of the time.
// Unfortunately, C++ will not do two-step conversions for us (future<future<R>> -> future<R> -> shared_future<R>).
// see: https://github.com/STEllAR-GROUP/hpx/issues/3758
template <typename R>
hpx::future<R> collapse(hpx::future<hpx::future<R>> &&fut)
{
    return { std::move(fut) };
}

template <typename R>
hpx::future<R> collapse(hpx::future<R> &&fut)
{
    return std::move(fut);
}

template <auto F, typename... Args>
decltype(auto)
named_make_tile(const tiled_scheduler_distributed & sched, std::size_t on, const char *name, Args &&...args)
{
    return collapse(hpx::dataflow(
        hpx::launch::async,
        hpx::annotated_function(hpx::unwrapping(typename plain_action_for<F>::action_type{}), name),
        sched.localities_[on],
        std::forward<Args>(args)...));
}

template <auto F, typename... Args>
decltype(auto)
named_dataflow(const tiled_scheduler_distributed &sched, std::size_t on, const char *name, Args &&...args)
{
    return collapse(hpx::dataflow(
        hpx::launch::async,
        hpx::annotated_function(hpx::unwrapping(typename plain_action_for<F>::action_type{}), name),
        sched.localities_[on],
        std::forward<Args>(args)...));
}

template <auto F, typename... Args>
decltype(auto) named_async(const tiled_scheduler_distributed &sched, std::size_t on, const char *name, Args &&...args)
{
    return hpx::async(hpx::annotated_function(typename plain_action_for<F>::action_type{}, name),
                      std::forward<Args>(args)...);
}

}  // namespace detail

GPRAT_NS_END

#endif
