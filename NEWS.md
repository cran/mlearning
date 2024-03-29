# mlearning 1.2.1

-   Documentation is refactored using Roxygen2 and considerably enhanced.

-   All camelCase function names now have their equivalence in snake_case, e.g., `mlRforest` -\> `ml_rforest()`, or `confusionImage()` -\> `confusion_image()` in order to adapt to the coding preferences of the user.

# mlearning 1.2.0

-   The `mlRpart()` function implements `rpart::rpart()` for using decision trees.

# mlearning 1.1.1

-   The description is extended.

-   A {pkgdown} site is added.

# mlearning 1.1.0

-   `mlKnn()` is implemented for K-nearest neighbors.

-   Several adjustments were required for compatibility with R 4.2.0 (it is not allowed any more to use vectors \> 1 with \|\| and &&).

# mlearning 1.0.7

-   When `predict()` was applied to an mlearning object build with full formula (not the short one `var ~ .`), if the dependent variable was not in `newdata =`, an error message was raised (although this variable is not necessary at this point). Bug identified by Damien Dumont, and corrected.

# mlearning 1.0.6

-   In `mlSvm.formula()`, arguments `scale=`, `type=`, `kernel=` and `classwt=` were not correctly used. Corrected.

# mlearning 1.0.5

-   In `mlLvq()` providing `size =` or `prior =` led to an `lvq` object not found message. Corrected.

# mlearning 1.0.4

-   Sometimes, data was not found (e.g., when called inside a {learnr} tutorial).

-   In `mlearning()`, data is forced with `as.data.frame()` (tibbles are not supported internally).

-   In the `mlXXX()` function, it was not possible to indicate something like `mlLda(data = iris, Species ~ .)`. Solved by adding `train =` argument in `mlXXX()`.

-   In `summary.confusion()` produced an error if more than one `type =` was provided.

# mlearning 1.0.3

-   NEWS.md file added. Repository moved to GitHub.
