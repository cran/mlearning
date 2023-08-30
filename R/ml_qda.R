#' Supervised classification using quadratic discriminant analysis
#'
#' @description
#' Unified (formula-based) interface version of the quadratic discriminant
#' analysis algorithm provided by [MASS::qda()].
#'
#' @param formula a formula with left term being the factor variable to predict
#' and the right term with the list of independent, predictive variables,
#' separated with a plus sign. If the data frame provided contains only the
#' dependent and independent variables, one can use the `class ~ .` short
#' version (that one is strongly encouraged). Variables with minus sign are
#' eliminated. Calculations on variables are possible according to usual
#' formula convention (possibly protected by using `I()`).
#' @param data a data.frame to use as a training set.
#' @param train a matrix or data frame with predictors.
#' @param response a vector of factor for the classification.
#' @param ... further arguments passed to [MASS::qda()] or its  [predict()]
#'   method (see the corresponding help page).
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_qda()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param object an **mlQda** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"membership"` the membership (a
#'   number between 0 and 1) to the different classes, or `"both"` to return
#'   classes and memberships.
#' @param prior the prior probabilities of class membership. By default, the
#'   prior are obtained from the object and, if they where not changed,
#'   correspond to the proportions observed in the training set.
#' @param method `"plug-in"`, `"predictive"`, `"debiased"`, `"looCV"`, or
#'   `"cv"`. `"plug-in"` (default) the usual unbiased parameter estimates are
#'   used. With `"predictive"`, the parameters are integrated out using a vague
#'   prior. With `"debiased"`, an unbiased estimator of the log posterior
#'   probabilities is used. With `"looCV"`, the leave-one-out cross-validation
#'   fits to the original data set are computed and returned. With `"cv"`,
#'   cross-validation is used instead. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case.
#'
#' @return [ml_qda()]/[mlQda()] creates an **mlQda**, **mlearning** object
#'   containing the classifier and a lot of additional metadata used by the
#'   functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also [MASS::qda()] that
#'   actually does the classification.
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
#' iris_qda <- ml_qda(data = iris_train, Species ~ .)
#' summary(iris_qda)
#' confusion(iris_qda)
#' confusion(predict(iris_qda, newdata = iris_test), iris_test$Species)
#'
#' # Another dataset (binary predictor... not optimal for qda, just for test)
#' data("HouseVotes84", package = "mlbench")
#' house_qda <- ml_qda(data = HouseVotes84, Class ~ ., na.action = na.omit)
#' summary(house_qda)
mlQda <- function(train, ...)
  UseMethod("mlQda")

#' @rdname mlQda
#' @export
ml_qda <- mlQda

#' @rdname mlQda
#' @export
#' @method mlQda formula
mlQda.formula <- function(formula, data, ..., subset, na.action) {
  mlearning(formula, data = data, method = "mlQda", model.args =
    list(formula  = formula, data = substitute(data),
      subset = substitute(subset)), call = match.call(), ...,
    subset = subset, na.action = substitute(na.action))
}

#' @rdname mlQda
#' @export
#' @method mlQda default
mlQda.default <- function(train, response, ...) {
  if (!is.factor(response))
    stop("only factor response (classification) accepted for mlQda")

  dots <- list(...)
  .args. <- dots$.args.
  dots$.args. <- NULL
  if (!length(.args.))
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = "classification", na.action = "na.pass",
      mlearning.call = match.call(), method = "mlQda")

  # Check if there are factor predictors
  if (any(sapply(train, is.factor)))
    warning("force conversion from factor to numeric; may be not optimal or suitable")

  # Return a mlearning object
  structure(MASS::qda(x = sapply(train, as.numeric),
    grouping = response, ...), formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = TRUE, type = .args.$type,
    pred.type = c(class = "class", membership = "posterior"),
    summary = NULL, na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "quadratic discriminant analysis",
    class = c("mlQda", "mlearning", "qda"))
}

#' @rdname mlQda
#' @export
#' @method predict mlQda
predict.mlQda <- function(object, newdata,
type = c("class", "membership", "both"), prior = object$prior,
method = c("plug-in", "predictive", "debiased", "looCV", "cv"), ...) {
  if (!inherits(object, "mlQda"))
    stop("'object' must be a 'mlQda' object")

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, prior = prior, ...))
  }

  # Recalculate newdata according to formula...
  if (missing(newdata)) {# Use train
    newdata <- attr(object, "train")
  } else if (attr(object, "optim")) {# Use optimized approach
    # Just keep vars similar as in train
    vars <- names(attr(object, "train"))
    if (!all(vars %in% names(newdata)))
      stop("One or more missing variables in newdata")
    newdata <- newdata[, vars]
  } else {# Use model.frame
    # but eliminate dependent variable, not required
    # (second item in the formula)
    newdata <- model.frame(formula = attr(object, "formula")[-2],
      data = newdata, na.action = na.pass)[, names(attr(object, "train"))]
  }
  # Only numerical predictors
  newdata <- sapply(as.data.frame(newdata), as.numeric)

  # Delegate to the MASS predict.qda method
  class(object) <- class(object)[-(1:2)]
  # I need to suppress warnings, because NAs produce ennoying warnings!
  if (method == "looCV") {
    res <- suppressWarnings(predict(object, prior = prior, method = method,
      ...))
  } else {
    res <- suppressWarnings(predict(object, newdata = newdata, prior = prior,
      method = method, ...))
  }

  # Rework results according to what we want
  switch(as.character(type)[1],
    class = factor(as.character(res$class), levels = levels(object)),
    membership = .membership(res$posterior, levels = levels(object)),
    both = list(class = factor(as.character(res$class),
      levels = levels(object)), membership = .membership(res$posterior,
        levels = levels(object))),
    stop("unrecognized 'type' (must be 'class', 'membership' or 'both')"))
}
