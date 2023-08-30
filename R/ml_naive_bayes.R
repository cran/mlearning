#' Supervised classification using naive Bayes
#'
#' @description
#' Unified (formula-based) interface version of the naive Bayes algorithm
#' provided by [e1071::naiveBayes()].
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
#' @param response a vector of factor with the classes.
#' @param laplace positive number controlling Laplace smoothing for the naive
#'   Bayes classifier. The default (0) disables Laplace smoothing.
#' @param ... further arguments passed to the classification method or its
#'   [predict()] method (not used here for now).
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_naive_bayes()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param object an **mlNaiveBayes** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"membership"`, the posterior
#'   probability or `"both"` to return classes and memberships,
#' @param method `"direct"` (default) or `"cv"`. `"direct"` predicts new cases in
#'   `newdata=` if this argument is provided, or the cases in the training set
#'   if not. Take care that not providing `newdata=` means that you just
#'   calculate the **self-consistency** of the classifier but cannot use the
#'   metrics derived from these results for the assessment of its performances.
#'   Either use a different dataset in `newdata=` or use the alternate
#'   cross-validation ("cv") technique. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case.
#' @param threshold value replacing cells with probabilities within 'eps' range.
#' @param eps number for specifying an epsilon-range to apply Laplace smoothing
#' (to replace zero or close-zero probabilities by 'threshold').
#'
#' @return [ml_naive_bayes()]/[mlNaiveBayes()] creates an **mlNaiveBayes**,
#'   **mlearning** object containing the classifier and a lot of additional
#'   metadata used by the functions and methods you can apply to it like
#'   [predict()] or [cvpredict()]. In case you want to program new functions or
#'   extract specific components, inspect the "unclassed" object using
#'   [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also
#'   [e1071::naiveBayes()] that actually does the classification.
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
#' iris_nb <- ml_naive_bayes(data = iris_train, Species ~ .)
#' summary(iris_nb)
#' predict(iris_nb) # Default type is class
#' predict(iris_nb, type = "membership")
#' predict(iris_nb, type = "both")
#' # Self-consistency, do not use for assessing classifier performances!
#' confusion(iris_nb)
#' # Use an independent test set instead
#' confusion(predict(iris_nb, newdata = iris_test), iris_test$Species)
#'
#' # Another dataset
#' data("HouseVotes84", package = "mlbench")
#' house_nb <- ml_naive_bayes(data = HouseVotes84, Class ~ .,
#'   na.action = na.omit)
#' summary(house_nb)
#' confusion(house_nb) # Self-consistency
#' confusion(cvpredict(house_nb), na.omit(HouseVotes84)$Class)
mlNaiveBayes <- function(train, ...)
  UseMethod("mlNaiveBayes")

#' @rdname mlNaiveBayes
#' @export
ml_naive_bayes <- mlNaiveBayes

#' @rdname mlNaiveBayes
#' @export
#' @method mlNaiveBayes formula
mlNaiveBayes.formula <- function(formula, data, laplace = 0, ...,
subset, na.action) {
  mlearning(formula, data = data, method = "mlNaiveBayes", model.args =
    list(formula  = formula, data = substitute(data),
      subset = substitute(subset)), call = match.call(), laplace = laplace,
    ..., subset = subset, na.action = substitute(na.action))
}

#' @rdname mlNaiveBayes
#' @export
#' @method mlNaiveBayes default
mlNaiveBayes.default <- function(train, response, laplace = 0, ...) {
  if (!is.factor(response))
    stop("only factor response (classification) accepted for mlNaiveBayes")

  dots <- list(...)
  .args. <- dots$.args.
  dots$.args. <- NULL
  dots$laplace <- laplace
  if (!length(.args.))
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = "classification", na.action = "na.pass",
      mlearning.call = match.call(), method = "mlNaiveBayes")

  # Return a mlearning object
  structure(e1071::naiveBayes(x = train, y = response,
    laplace = laplace, ...), formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = FALSE, type = .args.$type,
    pred.type = c(class = "class", membership = "raw"),
    summary = NULL, na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "naive Bayes classifier",
    class = c("mlNaiveBayes", "mlearning", "naiveBayes"))
}

#' @rdname mlNaiveBayes
#' @export
#' @method predict mlNaiveBayes
predict.mlNaiveBayes <- function(object, newdata,
  type = c("class", "membership", "both"), method = c("direct", "cv"),
  na.action = na.exclude, threshold = 0.001, eps = 0, ...) {
  if (!inherits(object, "mlNaiveBayes"))
    stop("'object' must be a 'mlNaiveBayes' object")

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, threshold = threshold,
      eps = eps, ...))
  }

  # Recalculate newdata according to formula...
  if (missing(newdata)) {# Use train
    newdata <- attr(object, "train")
  } else if (attr(object, "optim")[1]) {# Use optimized approach
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

  # Delegate to the e1071::predict.naiveBayes() method
  type <- as.character(type)[1]
  class(object) <- class(object)[-(1:2)]
  # This is for classification
  if (type == "membership" || type == "both")
    proba <- predict(object, newdata = newdata, type = "raw",
      threshold = threshold, eps = eps, ...)
  if (type == "class" || type == "both")
    res <- predict(object, newdata = newdata, type = "class",
      threshold = threshold, eps = eps, ...)

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
    stop("unrecognized 'type' (must be 'class', 'membership' or 'both')"))
}


## NaiveBayes from RWeka package
## TODO: keep this for mlearningWeka package!
#mlNaiveBayesWeka <- function (train, ...)
#  UseMethod("mlNaiveBayesWeka")
#
#mlNaiveBayesWeka.formula <- function(formula, data, ..., subset, na.action)
#  mlearning(formula, data = data, method = "mlNaiveBayesWeka", model.args =
#    list(formula  = formula, data = substitute(data),
#    subset = substitute(subset)), call = match.call(),
#    ..., subset = subset, na.action = substitute(na.action))
#
#mlNaiveBayesWeka.default <- function (train, response, ...)
#{
#  if (!is.factor(response))
#    stop("only factor response (classification) accepted for mlNaiveBayesWeka")
#
#  .args. <- dots <- list(...)$.args.
#  if (!length(.args.)) .args. <- list(levels = levels(response),
#    n = c(intial = NROW(train), final = NROW(train)),
#    type = "classification", na.action = "na.pass",
#    mlearning.call = match.call(), method = "mlNaiveBayesWeka")
#
#  wekaArgs <- list(control = .args.$control)
#
#  ## If response is not NULL, add it to train
#  if (length(response)) {
#    formula <- .args.$formula
#    if (!length(formula)) response.label <- "Class" else
#      response.label <- all.vars(formula)[1]
#    data <- data.frame(response, train)
#    names(data) <- c(response.label, colnames(train))
#    wekaArgs$data <- data
#    wekaArgs$formula <- as.formula(paste(response.label, "~ ."))
#  } else { # Unsupervised classification
#    wekaArgs$data <- train
#    wekaArgs$formula <- ~ .
#  }
#
#  WekaClassifier <- make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
#
#  ## Return a mlearning object
#  structure(do.call(WekaClassifier, wekaArgs), formula = .args.$formula,
#    train = train, response = response, levels = .args.$levels, n = .args.$n,
#    args = dots, optim = .args.$optim, numeric.only = FALSE,
#    type = .args.$type, pred.type = c(class = "class", membership = "probability"),
#    summary = "summary", na.action = .args.$na.action,
#    mlearning.call = .args.$mlearning.call, method = .args.$method,
#    algorithm = "Weka naive Bayes classifier",
#    class = c("mlNaiveBayesWeka", "mlearning", "Weka_classifier"))
#}

