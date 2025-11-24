using LinearAlgebra
using Plots

function simulate_population()
    # Define the matrix A
    A = [
        (0.9980 * 0.9413) + (0.634 / 1000) + (6.3 / 1000 * 0.01)    (6.3 / 1000 * 0.75)          (6.3 / 1000 * 0.24)          0  0  0;
        (1 - 0.9413) * 0.9980                                      (0.9448 * 0.9920) + (1.585 / 1000)  0  0  0  0;
        0                                                           (1 - 0.9448) * 0.9920      (0.9446 * 0.9567) + (0.761 / 1000)  0  0  0;
        0  0                                                        (1 - 0.9446) * 0.9567      (0.9663 * 0.7401) + (0.190 / 1000)  0  0;
        0  0  0                                                     (1 - 0.9663) * 0.7401      (0.9520 * 0.5806)  0;
        0  0  0  0                                                  (1 - 0.9520) * 0.5806      0
    ]

    # Initial population vector
    x0 = (205 / 100) .* [17140, 21253, 29983, 23921, 6599, 1105]

    years = 40
    x = x0

    total_population = Float64[]
    history = zeros(6, years+1)  # store x(k) for each year

    for k in 0:years
        history[:, k+1] = x
        push!(total_population, sum(x))
        x = A * x
    end

    return total_population, history
end


# Run simulation
total_population, history = simulate_population()

years = 0:40

# -------------------------------
# FIGURE 1: Total population
# -------------------------------
plot(years, total_population,
    xlabel="Year",
    ylabel="Total Population",
    title="Population Over 40 Years",
    legend=false
)
savefig("population_over_40_years.png")


# -------------------------------
# FIGURE 2: Each state x_i(k)
# -------------------------------
plot()
for i in 1:6
    plot!(years, history[i, :], label="x$i")
end

xlabel!("Year")
ylabel!("Population")
title!("State Populations Over 40 Years")

savefig("state_populations_over_40_years.png")
