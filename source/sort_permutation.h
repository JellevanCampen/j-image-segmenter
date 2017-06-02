#include <vector>
#include <numeric>

// Algorithms for finding a permutation that would sort a vector. Used to sort
// multiple vectors based on the order of a single vector. 
// Source: https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of

// Find the permutation required to sort a vector
template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
  const std::vector<T>& vec,
  Compare& compare)
{
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
    [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
  return p;
}

// Apply a permutation to a vector
template <typename T>
void apply_permutation_in_place(
  std::vector<T>& vec,
  const std::vector<std::size_t>& p)
{
  std::vector<bool> done(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i)
  {
    if (done[i])
    {
      continue;
    }
    done[i] = true;
    std::size_t prev_j = i;
    std::size_t j = p[i];
    while (i != j)
    {
      std::swap(vec[prev_j], vec[j]);
      done[j] = true;
      prev_j = j;
      j = p[j];
    }
  }
}