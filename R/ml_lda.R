# Note: ldahist() in MASS (when only one LD) seems to be broken!

#' Supervised classification using linear discriminant analysis
#'
#' @description
#' Unified (formula-based) interface version of the linear discriminant
#' analysis algorithm provided by [MASS::lda()].
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
#' @param ... further arguments passed to [MASS::lda()] or its  [predict()]
#'   method (see the corresponding help page).
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_lda()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param object an **mlLda** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"membership"` the membership (a
#'   number between 0 and 1) to the different classes, or `"both"` to return
#'   classes and memberships. The `type = "projection"` returns a projection
#'   of the individuals in the plane represented by the `dimension= `
#'   discriminant components.
#' @param prior the prior probabilities of class membership. By default, the
#'   prior are obtained from the object and, if they where not changed,
#'   correspond to the proportions observed in the training set.
#' @param dimension the number of the predictive space to use. If `NULL` (the
#'   default) a reasonable value is used. If this is less than min(p, ng-1),
#'   only the first `dimension` discriminant components are used (except for
#'   `method = "predictive"`), and only those dimensions are returned in x.
#' @param method `"plug-in"`, `"predictive"`, `"debiased"`, or `"cv"`.
#'   `"plug-in"` (default) the usual unbiased parameter estimates are used.
#'   With `"predictive"`, the parameters are integrated out using a vague prior.
#'   With `"debiased"`, an unbiased estimator of the log posterior probabilities
#'   is used. With `"cv"`, cross-validation is used instead. If you specify
#'   `method = "cv"` then [cvpredict()] is used and you cannot provide
#'   `newdata=` in that case.
#'
#' @return [ml_lda()]/[mlLda()] creates an **mlLda**, **mlearning** object
#'   containing the classifier and a lot of additional metadata used by the
#'   functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also [MASS::lda()] that
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
#' iris_lda <- ml_lda(data = iris_train, Species ~ .)
#' iris_lda
#' summary(iris_lda)
#' plot(iris_lda, col = as.numeric(response(iris_lda)) + 1)
#' # Prediction using a test set
#' predict(iris_lda, newdata = iris_test) # class (default type)
#' predict(iris_lda, type = "membership") # posterior probability
#' predict(iris_lda, type = "both") # both class and membership in a list
#' # Type projection
#' predict(iris_lda, type = "projection") # Projection on the LD axes
#' # Add test set items to the previous plot
#' points(predict(iris_lda, newdata = iris_test, type = "projection"),
#'   col = as.numeric(predict(iris_lda, newdata = iris_test)) + 1, pch = 19)
#' # predict() and confusion() should be used on a separate test set
#' # for unbiased estimation (or using cross-validation, bootstrap, ...)
#' # Wrong, cf. biased estimation (so-called, self-consistency)
#' confusion(iris_lda)
#' # Estimation using a separate test set
#' confusion(predict(iris_lda, newdata = iris_test), iris_test$Species)
#'
#' # Another dataset (binary predictor... not optimal for lda, just for test)
#' data("HouseVotes84", package = "mlbench")
#' house_lda <- ml_lda(data = HouseVotes84, na.action = na.omit, Class ~ .)
#' summary(house_lda)
#' confusion(house_lda) # Self-consistency (biased metrics)
#' print(confusion(house_lda), error.col = FALSE) # Without error column
#'
#' # More complex formulas
#' # Exclude one or more variables
#' iris_lda2 <- ml_lda(data = iris, Species ~ . - Sepal.Width)
#' summary(iris_lda2)
#' # With calculation
#' iris_lda3 <- ml_lda(data = iris, Species ~ log(Petal.Length) +
#'   log(Petal.Width) + I(Petal.Length/Sepal.Length))
#' summary(iris_lda3)
#'
#' # Factor levels with missing items are allowed
#' ir2 <- iris[-(51:100), ] # No Iris versicolor in the training set
#' iris_lda4 <- ml_lda(data = ir2, Species ~ .)
#' summary(iris_lda4) # missing class
#' # Missing levels are reinjected in class or membership by predict()
#' predict(iris_lda4, type = "both")
#' # ... but, of course, the classifier is wrong for Iris versicolor
#' confusion(predict(iris_lda4, newdata = iris), iris$Species)
#'
#' # Simpler interface, but more memory-effective
#' iris_lda5 <- ml_lda(train = iris[, -5], response = iris$Species)
#' summary(iris_lda5)
mlLda <- function(train, ...)
  UseMethod("mlLda")

#' @rdname mlLda
#' @export
ml_lda <- mlLda

#' @rdname mlLda
#' @export
#' @method mlLda formula
mlLda.formula <- function(formula, data, ..., subset, na.action)
  mlearning(formula, data = data, method = "mlLda", model.args =
      list(formula  = formula, data = substitute(data),
        subset = substitute(subset)), call = match.call(), ...,
    subset = subset, na.action = substitute(na.action))

#' @rdname mlLda
#' @export
#' @method mlLda default
mlLda.default <- function(train, response, ...) {
  if (!is.factor(response))
    stop("only factor response (classification) accepted for mlLda")

  dots <- list(...)
  .args. <- dots$.args.
  dots$.args. <- NULL
  if (!length(.args.))
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = "classification", na.action = "na.pass",
      mlearning.call = match.call(), method = "mlLda")

  # Check if there are factor predictors
  if (any(sapply(train, is.factor)))
    warning("force conversion from factor to numeric; may be not optimal or suitable")

  # Return a mlearning object
  structure(MASS::lda(x = sapply(train, as.numeric),
    grouping = response, ...), formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = TRUE, type = .args.$type,
    pred.type = c(class = "class", membership = "posterior", projection = "x"),
    summary = NULL, na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "linear discriminant analysis",
    class = c("mlLda", "mlearning", "lda"))
}

#' @rdname mlLda
#' @export
#' @method predict mlLda
predict.mlLda <- function(object, newdata,
type = c("class", "membership", "both", "projection"), prior = object$prior,
dimension = NULL, method = c("plug-in", "predictive", "debiased", "cv"), ...) {
  if (!inherits(object, "mlLda"))
    stop("'object' must be a 'mlLda' object")

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    if (is.null(dimension)) {
      return(cvpredict(object = object, type = type, prior = prior, ...))
    } else {
      return(cvpredict(object = object, type = type, prior = prior,
        dimension = dimension, ...))
    }
  }

  # Recalculate newdata according to formula...
  if (missing(newdata)) {# Use train
    newdata <- attr(object, "train")
  } else if (attr(object, "optim")) {# Use optimized approach
    ## Just keep vars similar as in train
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

  # dimension
  if (missing(dimension)) {
    dimension <- length(object$svd)
  } else {
    dimension <- min(dimension, length(object$svd))
  }

  # Delegate to the MASS predict.lda method
  class(object) <- class(object)[-(1:2)]
  # I need to suppress warnings, because NAs produce ennoying warnings!
  res <- suppressWarnings(predict(object, newdata = newdata, prior = prior,
    dimen = dimension, method = method, ...))

  # Rework results according to what we want
  switch(as.character(type)[1],
    class = factor(as.character(res$class), levels = levels(object)),
    membership = .membership(res$posterior, levels = levels(object)),
    both = list(class = factor(as.character(res$class),
      levels = levels(object)), membership = .membership(res$posterior,
        levels = levels(object))),
    projection = res$x,
    stop("unrecognized 'type' (must be 'class', 'membership', 'both' or 'projection')"))
}
