#' Supervised classification using learning vector quantization
#'
#' @description
#' Unified (formula-based) interface version of the learning vector quantization
#' algorithms provided by [class::olvq1()], [class::lvq1()], [class::lvq2()],
#' and [class::lvq3()].
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
#' @param response a vector of factor of the classes.
#' @param k.nn k used for k-NN number of neighbor considered. Default is 5.
#' @param size the size of the codebook. Defaults to
#'   min(round(0.4 \* nc \* (nc - 1 + p/2),0), n) where nc is the number of
#'   classes.
#' @param prior probabilities to represent classes in the codebook (default
#'   values are the proportions in the training set).
#' @param algorithm `"olvq1"` (by default, the optimized 'lvq1' version), or
#' `"lvq1"`, `"lvq2"`, `"lvq3"`.
#' @param ... further arguments passed to the classification method or its
#'   [predict()] method (not used here for now).
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_lvq)] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param x,object an **mlLvq** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param type the type of prediction to return. For this method, only `"class"`
#'   is accepted, and it is the default. It returns the predicted classes.
#' @param method `"direct"` (default) or `"cv"`. `"direct"` predicts new cases in
#'   `newdata=` if this argument is provided, or the cases in the training set
#'   if not. Take care that not providing `newdata=` means that you just
#'   calculate the **self-consistency** of the classifier but cannot use the
#'   metrics derived from these results for the assessment of its performances.
#'   Either use a different dataset in `newdata=` or use the alternate
#'   cross-validation ("cv") technique. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case.
#'
#' @return [ml_lvq()]/[mlLvq()] creates an **mlLvq**, **mlearning** object
#'   containing the classifier and a lot of additional metadata used by the
#'   functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also [class::olvq1()],
#'   [class::lvq1()], [class::lvq2()], and [class::lvq3()] that actually do the
#'    classification.
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
#' iris_lvq <- ml_lvq(data = iris_train, Species ~ .)
#' summary(iris_lvq)
#' predict(iris_lvq) # This object only returns classes
#' #' # Self-consistency, do not use for assessing classifier performances!
#' confusion(iris_lvq)
#' # Use an independent test set instead
#' confusion(predict(iris_lvq, newdata = iris_test), iris_test$Species)
mlLvq <- function(train, ...)
  UseMethod("mlLvq")

#' @rdname mlLvq
#' @export
ml_lvq <- mlLvq

#' @rdname mlLvq
#' @export
#' @method mlLvq formula
mlLvq.formula <- function(formula, data, k.nn = 5, size, prior,
  algorithm = "olvq1", ..., subset, na.action) {
  if (missing(size)) {
    if (missing(prior)) {
      mlearning(formula, data = data, method = "mlLvq", model.args =
          list(formula  = formula, data = substitute(data),
            subset = substitute(subset)), call = match.call(), k.nn = k.nn,
        algorithm = algorithm, ...,
        subset = subset, na.action = substitute(na.action))
    } else {
      mlearning(formula, data = data, method = "mlLvq", model.args =
          list(formula  = formula, data = substitute(data),
            subset = substitute(subset)), call = match.call(), k.nn = k.nn,
        prior = prior, algorithm = algorithm, ...,
        subset = subset, na.action = substitute(na.action))
    }
  } else {
    if (missing(prior)) {
      mlearning(formula, data = data, method = "mlLvq", model.args =
          list(formula  = formula, data = substitute(data),
            subset = substitute(subset)), call = match.call(), k.nn = k.nn,
        size = size, algorithm = algorithm, ...,
        subset = subset, na.action = substitute(na.action))
    } else {
      mlearning(formula, data = data, method = "mlLvq", model.args =
          list(formula  = formula, data = substitute(data),
            subset = substitute(subset)), call = match.call(), k.nn = k.nn,
        size = size, prior = prior, algorithm = algorithm, ...,
        subset = subset, na.action = substitute(na.action))
    }
  }
}

#' @rdname mlLvq
#' @export
#' @method mlLvq default
mlLvq.default <- function(train, response, k.nn = 5, size, prior,
  algorithm = "olvq1", ...) {
  if (!is.factor(response))
    stop("only factor response (classification) accepted for mlLvq")

  dots <- list(...)
  .args. <- dots$.args.
  dots$.args. <- NULL
  dots$k.nn <- k.nn
  dots$algorithm <- algorithm
  if (!length(.args.))
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = "classification", na.action = "na.pass",
      mlearning.call = match.call(), method = "mlLvq")

  # Matrix of numeric values
  if (any(sapply(train, is.factor))) {
    warning("force conversion from factor to numeric; may be not optimal or suitable")
    train <- sapply(train, as.numeric)
  }

  # Default values for size and prior, if not provided
  n <- nrow(train)
  if (missing(prior) || !length(prior)) {
    prior <- tapply(rep(1, length(response)), response, sum) / n
  } else dots$prior <- prior
  if (missing(size) || !length(size)) {
    np <- length(prior)
    size <- min(round(0.4 * np * (np - 1 + ncol(train) / 2), 0), n)
  } else dots$size <- size

  # Initialize codebook
  init <- lvqinit(train, response, k = k.nn, size = size, prior = prior)

  # Calculate final codebook
  if (algorithm[1] == "olvq1") times <- 40 else times <- 100
  niter <- dots$niter
  if (!length(niter)) niter <- times * nrow(init$x) # Default value
  alpha <- dots$alpha
  if (!length(alpha)) alpha <- if (algorithm[1] == "olvq1") 0.3 else 0.03
  win <- dots$win
  if (!length(win)) win <- 0.3
  epsilon <- dots$epsilon
  if (!length(epsilon)) epsilon <- 0.1
  codebk <- switch(algorithm,
    olvq1 = class::olvq1(train, response, init, niter = niter,
      alpha = alpha),
    lvq1 = class::lvq1(train, response, init, niter = niter,
      alpha = alpha),
    lvq2 = class::lvq2(train, response, init, niter = niter,
      alpha = alpha, win = win),
    lvq3 = class::lvq3(train, response, init, niter = niter,
      alpha = alpha, win = win, epsilon = epsilon),
    stop("algorithm must be 'lvq1', 'lvq2', 'lvq3' or 'olvq1'"))

  # Return a mlearning object
  structure(codebk, formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = TRUE, type = .args.$type,
    pred.type = c(class = "class"), summary = "summary.lvq",
    na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "learning vector quantization",
    class = c("mlLvq", "mlearning", class(codebk)))
}

#' @rdname mlLvq
#' @export
#' @method summary mlLvq
summary.mlLvq <- function(object, ...)
  structure(cbind(Class = object$cl, as.data.frame(object$x)),
    class = c("summary.mlLvq", "data.frame"))

#' @rdname mlLvq
#' @export
#' @method print summary.mlLvq
print.summary.mlLvq <- function(x, ...) {
  cat("Codebook:\n")
  print(as.data.frame(x))
  invisible(x)
}

#' @rdname mlLvq
#' @export
#' @method predict mlLvq
predict.mlLvq <- function(object, newdata, type = "class",
  method = c("direct", "cv"), na.action = na.exclude, ...) {
  if (!inherits(object, "mlLvq"))
    stop("'object' must be a 'mlLvq' object")
  if (type != "class")
    stop("Only 'class' currently supported for type")

  # If method == "cv", delegate to cvpredict()
  if (as.character(method)[1] == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, ...))
  }

  # Recalculate newdata according to formula...
  if (missing(newdata)) { # Use train
    newdata <- attr(object, "train")
  } else if (attr(object, "optim")) {# Use optimized approach
    ## Just keep vars similar as in train
    vars <- colnames(attr(object, "train"))
    if (!all(vars %in% names(newdata)))
      stop("one or more missing variables in newdata")
    newdata <- newdata[, vars]
  } else {# Use model.frame
    # but eliminate dependent variable, not required
    # (second item in the formula)
    newdata <- model.frame(formula = attr(object, "formula")[-2],
      data = newdata, na.action = na.pass)[, names(attr(object, "train"))]
  }
  newdata <- sapply(as.data.frame(newdata), as.numeric)

  # Determine how many data and perform na.action
  n <- NROW(newdata)
  newdata <- match.fun(na.action)(newdata)
  ndrop <- attr(newdata, "na.action")

  .expandFactor(lvqtest(object, newdata), n, ndrop)
}
