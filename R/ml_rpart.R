#' Supervised classification and regression using recursive partitioning
#'
#' @description
#' Unified (formula-based) interface version of the recursive partitioning
#' algorithm as implemented in [rpart::rpart()].
#'
#' @param formula a formula with left term being the factor variable to predict
#' (for supervised classification), a vector of numbers (for regression) and the
#' right term with the list of independent, predictive variables, separated with
#' a plus sign. If the data frame provided contains only the dependent and
#' independent variables, one can use the `class ~ .` short version (that one is
#' strongly encouraged). Variables with minus sign are eliminated. Calculations
#' on variables are possible according to usual formula convention (possibly
#' protected by using `I()`).
#' @param data a data.frame to use as a training set.
#' @param train a matrix or data frame with predictors.
#' @param response a vector of factor (classification) or numeric (regression).
#' @param ... further arguments passed to [rpart::rpart()] or its [predict()]
#'   method (see the corresponding help page.
#' @param .args. used internally, do not provide anything here.
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_rpart()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param object an **mlRpart** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"membership"` the membership (number
#'   between 0 and 1) to the different classes, or `"both"` to return classes
#'   and memberships,
#' @param method `"direct"` (default) or `"cv"`. `"direct"` predicts new cases in
#'   `newdata=` if this argument is provided, or the cases in the training set
#'   if not. Take care that not providing `newdata=` means that you just
#'   calculate the **self-consistency** of the classifier but cannot use the
#'   metrics derived from these results for the assessment of its performances.
#'   Either use a different data set in `newdata=` or use the alternate
#'   cross-validation ("cv") technique. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case.
#'
#' @return [ml_rpart()]/[mlRpart()] creates an **mlRpart**, **mlearning** object
#'   containing the classifier and a lot of additional metadata used by the
#'   functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also [rpart::rpart()]
#'   that actually does the classification.
#' @export
#'
#' @examples
#' # Prepare data: split into training set (2/3) and test set (1/3)
#' data("iris", package = "datasets")
#' train <- c(1:34, 51:83, 101:133)
#' iris_train <- iris[train, ]
#' iris_test <- iris[-train, ]
#' # One case with missing data in train set, and another case in test set
#' iris_train[1, 1] <- NA
#' iris_test[25, 2] <- NA
#'
#' iris_rpart <- ml_rpart(data = iris_train, Species ~ .)
#' summary(iris_rpart)
#' # Plot the decision tree for this classifier
#' plot(iris_rpart, margin = 0.03, uniform = TRUE)
#' text(iris_rpart, use.n = FALSE)
#' # Predictions
#' predict(iris_rpart) # Default type is class
#' predict(iris_rpart, type = "membership")
#' predict(iris_rpart, type = "both")
#' # Self-consistency, do not use for assessing classifier performances!
#' confusion(iris_rpart)
#' # Cross-validation prediction is a good choice when there is no test set
#' predict(iris_rpart, method = "cv")  # Idem: cvpredict(res)
#' confusion(iris_rpart, method = "cv")
#' # Evaluation of performances using a separate test set
#' confusion(predict(iris_rpart, newdata = iris_test), iris_test$Species)
mlRpart <- function(train, ...)
  UseMethod("mlRpart")

#' @rdname mlRpart
#' @export
ml_rpart <- mlRpart

#' @rdname mlRpart
#' @export
#' @method mlRpart formula
mlRpart.formula <- function(formula, data, ..., subset, na.action) {
  mlearning(formula, data = data, method = "mlRpart", model.args =
      list(formula  = formula, data = substitute(data),
        subset = substitute(subset)), call = match.call(), ...,
    subset = subset, na.action = substitute(na.action))
}

#' @rdname mlRpart
#' @export
#' @method mlRpart default
mlRpart.default <- function(train, response, ..., .args. = NULL) {
  dots <- list(...)
  if (is.null(.args.) || !length(.args.)) {
    if (!length(response)) {# unsupervised
      stop("Unsupervised classification is not possible with mlRpart(), see hclust() instead.")
    } else if (is.factor(response)) {
      type <- "classification"
    } else type <- "regression"

    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = type, na.action = "na.pass",
      mlearning.call = match.call(), method = "mlRpart")
  }

  # Combine train + response and use the formula interface which is the only one
  data <- train
  data$.class <- response
  res <- rpart::rpart(.class ~ ., data = data, ...)
  res$predicted <- predict(res, type = "class")

  # Return a mlearning object
  structure(res, formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = FALSE, type = .args.$type,
    pred.type = c(class = "class", membership = "prob"),
    summary = NULL, na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "recursive partitioning tree",
    class = c("mlRpart", "mlearning", "rpart"))
}

#' @rdname mlRpart
#' @export
#' @method predict mlRpart
predict.mlRpart <- function(object, newdata,
  type = c("class", "membership", "both"),
  method = c("direct", "cv"), ...) {

  type <- as.character(type)[1]

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, ...))
  } else {
    predict.mlearning(object = object, newdata = newdata,
      type = type, ...)
  }
}
