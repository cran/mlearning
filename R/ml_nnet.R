#' Supervised classification and regression using neural network
#'
#' @description
#' Unified (formula-based) interface version of the single-hidden-layer neural
#' network algorithm, possibly with skip-layer connections provided by
#' [nnet::nnet()].
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
#' @param size number of units in the hidden layer. Can be zero if there are
#'   skip-layer units. If `NULL` (the default), a reasonable value is computed.
#' @param rang initial random weights on \[-rang, rang\]. Value about 0.5 unless
#'   the inputs are large, in which case it should be chosen so that
#'   rang * max(|x|) is about 1. If `NULL`, a reasonable default is computed.
#' @param decay parameter for weight decay. Default to 0.
#' @param maxit maximum number of iterations. Default 1000 (it is 100 in
#'   [nnet::nnet()]).
#' @param ... further arguments passed to [nnet::nnet()] that has many more
#'   parameters (see its help page).
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_nnet()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param object an **mlNnet** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"membership"` the membership (number
#'   between 0 and 1) to the different classes, or `"both"` to return classes
#'   and memberships. Also type `"raw"` as non normalized result as returned by
#'   [nnet::nnet()] (useful for regression, see examples).
#' @param method `"direct"` (default) or `"cv"`. `"direct"` predicts new cases
#'   in `newdata=` if this argument is provided, or the cases in the training
#'   set if not. Take care that not providing `newdata=` means that you just
#'   calculate the **self-consistency** of the classifier but cannot use the
#'   metrics derived from these results for the assessment of its performances.
#'   Either use a different data set in `newdata=` or use the alternate
#'   cross-validation ("cv") technique. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case.
#'
#' @return [ml_nnet()]/[mlNnet()] creates an **mlNnet**, **mlearning** object
#'   containing the classifier and a lot of additional metadata used by the
#'   functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also [nnet::nnet()]
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
#' set.seed(689) # Useful for reproductibility, use a different value each time!
#' iris_nnet <- ml_nnet(data = iris_train, Species ~ .)
#' summary(iris_nnet)
#' predict(iris_nnet) # Default type is class
#' predict(iris_nnet, type = "membership")
#' predict(iris_nnet, type = "both")
#' # Self-consistency, do not use for assessing classifier performances!
#' confusion(iris_nnet)
#' # Use an independent test set instead
#' confusion(predict(iris_nnet, newdata = iris_test), iris_test$Species)
#'
#' # Idem, but two classes prediction
#' data("HouseVotes84", package = "mlbench")
#' set.seed(325)
#' house_nnet <- ml_nnet(data = HouseVotes84, Class ~ ., na.action = na.omit)
#' summary(house_nnet)
#' # Cross-validated confusion matrix
#' confusion(cvpredict(house_nnet), na.omit(HouseVotes84)$Class)
#'
#' # Regression
#' data(airquality, package = "datasets")
#' set.seed(74)
#' ozone_nnet <- ml_nnet(data = airquality, Ozone ~ ., na.action = na.omit,
#'   skip = TRUE, decay = 1e-3, size = 20, linout = TRUE)
#' summary(ozone_nnet)
#' plot(na.omit(airquality)$Ozone, predict(ozone_nnet, type = "raw"))
#' abline(a = 0, b = 1)
mlNnet <- function(train, ...)
  UseMethod("mlNnet")

#' @rdname mlNnet
#' @export
ml_nnet <- mlNnet

#' @rdname mlNnet
#' @export
#' @method mlNnet formula
mlNnet.formula <- function(formula, data, size = NULL, rang = NULL, decay = 0,
maxit = 1000, ..., subset, na.action) {
  mlearning(formula, data = data, method = "mlNnet", model.args =
    list(formula  = formula, data = substitute(data),
      subset = substitute(subset)), call = match.call(), size = size,
    rang = rang, decay = decay, maxit = maxit, ...,
    subset = subset, na.action = substitute(na.action))
}

#' @rdname mlNnet
#' @export
#' @method mlNnet default
mlNnet.default <- function(train, response, size = NULL, rang = NULL, decay = 0,
maxit = 1000, ...) {
  if (!length(response))
    stop("unsupervised classification not usable for mlNnet")

  nnetArgs <- dots <- list(...)
  .args. <- nnetArgs$.args.
  dots$.args. <- NULL
  dots$size <- size
  dots$rang <- rang
  dots$decay <- decay
  dots$maxit <- maxit
  nnetArgs$.args. <- NULL
  if (!length(.args.))
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = if (is.factor(response)) "classification" else "regression",
      na.action = "na.pass", mlearning.call = match.call(), method = "mlNnet")

  # Construct arguments list for nnet() call
  nnetArgs$x <- sapply(train, as.numeric)

  # Weights
  if (!length(nnetArgs$weights))
    nnetArgs$weights <- .args.$weights

  # size
  if (!length(size))
    size <- length(levels(response)) - 1 # Is this a reasonable default?
  nnetArgs$size <- size

  # rang
  if (!length(rang)) {
    # default is 0.7 in original nnet code,
    # but the doc proposes something else
    rang <- round(1 / max(abs(nnetArgs$x)), 2)
    if (rang < 0.01) rang <- 0.01
    if (rang > 0.7) rang <- 0.7
  }
  nnetArgs$rang <- rang

  # decay and maxit
  nnetArgs$decay <- decay
  nnetArgs$maxit <- maxit

  # TODO: should I need to implement this???
  #x <- model.matrix(Terms, m, contrasts)
  #cons <- attr(x, "contrast")
  #xint <- match("(Intercept)", colnames(x), nomatch = 0L)
  #if (xint > 0L)
  #    x <- x[, -xint, drop = FALSE]

  # Classification or regression?
  if (is.factor(response)) {
    if (length(levels(response)) == 2L) {
      nnetArgs$y <- as.vector(unclass(response)) - 1
      nnetArgs$entropy <- TRUE
      res <- do.call(nnet.default, nnetArgs)
      res$lev <- .args.$levels
    } else {
      nnetArgs$y <- nnet::class.ind(response)
      nnetArgs$softmax <- TRUE
      res <- do.call(nnet.default, nnetArgs)
      res$lev <- .args.$levels
    }
  } else {# Regression
    nnetArgs$y <- response
    res <- do.call(nnet.default, nnetArgs)
  }

  # Return a mlearning object
  structure(res, formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = TRUE, type = .args.$type,
    pred.type = c(class = "class", membership = "raw"),
    summary = "summary", na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "single-hidden-layer neural network",
    class = c("mlNnet", "mlearning", "nnet"))
}

#' @rdname mlNnet
#' @export
#' @method predict mlNnet
predict.mlNnet <- function(object, newdata,
type = c("class", "membership", "both", "raw"), method = c("direct", "cv"),
na.action = na.exclude, ...) {
  if (!inherits(object, "mlNnet"))
    stop("'object' must be a 'mlNnet' object")

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, ...))
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

  # Determine how many data and perform na.action
  n <- NROW(newdata)
  newdata <- match.fun(na.action)(newdata)
  ndrop <- attr(newdata, "na.action")
  attr(newdata, "na.action") <- NULL

  # Delegate to the nnet predict.nnet() method
  type <- as.character(type)[1]
  class(object) <- class(object)[-(1:2)]
  # This is for classification
  if (type == "membership" || type == "both")
    proba <- predict(object, newdata = newdata, type = "raw", ...)
  if (type == "class" || type == "both")
    res <- predict(object, newdata = newdata, type = "class", ...)
  if (type == "raw")
    res <- predict(object, newdata = newdata, type = "raw", ...)

  # Rework results according to what we want
  switch(type,
    class = .expandFactor(factor(as.character(res),
      levels = levels(object)), n, ndrop),
    membership = .expandMatrix(.membership(proba,
      levels = levels(object)), n, ndrop),
    both = list(class = .expandFactor(factor(as.character(res),
      levels = levels(object)), n, ndrop),
      membership = .expandMatrix(.membership(proba,
        levels = levels(object)), n, ndrop)),
    raw = res,
    stop("unrecognized 'type' (must be 'class', 'membership', 'both' or 'raw')")
  )
}
