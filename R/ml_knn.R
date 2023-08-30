#' Supervised classification using k-nearest neighbor
#'
#' @description
#' Unified (formula-based) interface version of the k-nearest neighbor
#' algorithm provided by [class::knn()].
#'
#' @param formula a formula with left term being the factor variable to predict
#' and the right term with the list of independent, predictive variables,
#' separated with a plus sign. If the data frame provided contains only the
#' dependent and independent variables, one can use the `class ~ .` short
#' version (that one is strongly encouraged). Variables with minus sign are
#' eliminated. Calculations on variables are possible according to usual formula
#' convention (possibly protected by using `I()`).
#' @param data a data.frame to use as a training set.
#' @param train a matrix or data frame with predictors.
#' @param response a vector of factor for the classification.
#' @param k.nn k used for k-NN number of neighbor considered. Default is 5.
#' @param ... further arguments passed to the classification method or its
#'   [predict()] method (not used here for now).
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_knn()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param x,object an **mlKnn** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"prob"` the "probability" for the
#'   different classes as assessed by the number of neighbors of these classes,
#'   or `"both"` to return classes and "probabilities",
#' @param method `"direct"` (default) or `"cv"`. `"direct"` predicts new cases in
#'   `newdata=` if this argument is provided, or the cases in the training set
#'   if not. Take care that not providing `newdata=` means that you just
#'   calculate the **self-consistency** of the classifier but cannot use the
#'   metrics derived from these results for the assessment of its performances.
#'   Either use a different data set in `newdata=` or use the alternate
#'   cross-validation ("cv") technique. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case.
#'
#' @return [ml_knn()]/[mlKnn()] creates an **mlKnn**, **mlearning** object
#'   containing the classifier and a lot of additional metadata used by the
#'   functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also [class::knn()] and
#'   [ipred::predict.ipredknn()] that actually do the classification.
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
#' iris_knn <- ml_knn(data = iris_train, Species ~ .)
#' summary(iris_knn)
#' predict(iris_knn) # This object only returns classes
#' # Self-consistency, do not use for assessing classifier performances!
#' confusion(iris_knn)
#' # Use an independent test set instead
#' confusion(predict(iris_knn, newdata = iris_test), iris_test$Species)
mlKnn <- function(train, ...)
  UseMethod("mlKnn")

#' @rdname mlKnn
#' @export
ml_knn <- mlKnn

#' @rdname mlKnn
#' @export
#' @method mlKnn formula
mlKnn.formula <- function(formula, data, k.nn = 5, ..., subset, na.action) {
  mlearning(formula, data = data, method = "mlKnn", model.args =
      list(formula  = formula, data = substitute(data),
        subset = substitute(subset)), call = match.call(), k.nn = k.nn, ...,
    subset = subset, na.action = substitute(na.action))
}

#' @rdname mlKnn
#' @export
#' @method mlKnn default
mlKnn.default <- function(train, response, k.nn = 5, ...) {
  if (!is.factor(response))
    stop("only factor response (classification) accepted for mlKnn")

  dots <- list(...)
  .args. <- dots$.args.
  dots$.args. <- NULL
  dots$k.nn <- k.nn
  if (!length(.args.))
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = "classification", na.action = "na.pass",
      mlearning.call = match.call(), method = "mlKnn")

  # matrix of numeric values
  if (any(sapply(train, is.factor))) {
    warning("force conversion from factor to numeric; may be not optimal or suitable")
    train <- sapply(train, as.numeric)
  }

  # Create an object similar to the one obtained with ipred::ipredknn
  res <- list(learn = list(y = response, X = train))
  res$k <- k.nn
  class(res) <- "simpleKnn"

  # Return a mlearning object
  structure(res, formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = TRUE, type = .args.$type,
    pred.type = c(class = "class", prob = "prob"), summary = NULL,
    na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "k-nearest neighbors",
    class = c("mlKnn", "mlearning", class(res)))
}

#' @rdname mlKnn
#' @export
#' @method summary mlKnn
summary.mlKnn <- function(object, ...)
  structure(cbind(Class = object$cl, as.data.frame(object$x)),
    class = c("summary.mlKnn", "data.frame"))

#' @rdname mlKnn
#' @export
#' @method print summary.mlKnn
print.summary.mlKnn <- function(x, ...) {
  cat("Train dataset:\n")
  print(as.data.frame(x))
  invisible(x)
}

#' @rdname mlKnn
#' @export
#' @method predict mlKnn
predict.mlKnn <- function(object, newdata,
  type = c("class", "prob", "both"),
  method = c("direct", "cv"), na.action = na.exclude, ...) {
  if (!inherits(object, "mlKnn"))
    stop("'object' must be a 'mlKnn' object")

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, na.action = na.action, ...))
  }

  # Recalculate newdata according to formula...
  if (missing(newdata))
    newdata <- object$learn$X # Use train
  # Use model.frame but eliminate dependent variable, not required
  # (second item in the formula)
  newdata <- model.frame(formula = attr(object, "formula")[-2],
    data = newdata, na.action = na.pass)[, names(object$learn$X)]
  # Only numerical predictors
  newdata <- sapply(as.data.frame(newdata), as.numeric)

  # Determine how many data and perform na.action
  n <- NROW(newdata)
  newdata <- match.fun(na.action)(newdata)
  ndrop <- attr(newdata, "na.action")

  if (inherits(object, "simpleKnn")) {
    res <- class::knn(object$learn$X, newdata, object$learn$y, k = object$k,
      prob = TRUE)
  } else {# ipred::ipredknn
    res <- ipred::predict.ipredknn(object, newdata, type = "class")
  }
  type <- as.character(type[1])
  if (type == "prob") {
    return(attr(res, "prob"))
  } else if (type == "class") {
    attr(res, "prob") <- NULL
  }

  .expandFactor(res, n, ndrop)
}
