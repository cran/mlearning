# TODO: test performances of optimized code for Class ~ .

#' Machine learning model for (un)supervised classification or regression
#'
#' @description
#' An **mlearning** object provides an unified (formula-based) interface to
#' several machine learning algorithms. They share the same interface and very
#' similar arguments. They conform to the formula-based approach, of say,
#' [stats::lm()] in base R, but with a coherent handling of missing data and
#' missing class levels. An optimized version exists for the simplified `y ~ .`
#' formula. Finally, cross-validation is also built-in.
#'
#' @param formula a formula with left term being the factor variable to predict
#' (for supervised classification), a vector of numbers (for regression) or
#' nothing (for unsupervised classification) and the right term with the list
#' of independent, predictive variables, separated with a plus sign. If the
#' data frame provided contains only the dependent and independent variables,
#' one can use the `class ~ .` short version (that one is strongly encouraged).
#' Variables with minus sign are eliminated. Calculations on variables are
#' possible according to usual formula convention (possibly protected by using
#' `I()`). Supervised classification, regression or unsupervised classification
#' are not available for all algorithms. Check respective help pages.
#' @param data a data.frame to use as a training set.
#' @param method the method to use. It can be `"mlLda"`, `"mlQda"`,
#'   `"mlNaiveBayes"`, `"mlKnn"`, `"mlLvq"`, `"mlNnet"`, `"mlRpart"`,
#'   `"mlRforest'`, or `"mlSvm"`. Respective functions exists as a shortcut,
#'   e.g., `mlLda()` is equivalent to `mlearning(method = "mlLda") for the
#'   formula interface. Users are supposed to call the shortcut functions. The
#'   `mlearning()` function serves mainly as core code for the various versions
#'   for developers.
#' @param model.args arguments for formula modeling with substituted data and
#'   subset... Not to be used by the end-user.
#' @param call the function call. Not to be used by the end-user.
#' @param ... further arguments (depends on the method).
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
#' @param x,object an **mlearning** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param y a second **mlearning** object or nothing (not used in several plots)
#' @param type the type of prediction to return. `"class"` by default, the
#'   predicted classes. Other options are `"membership"` the membership (a
#'   number between 0 and 1) to the different classes, or `"both"` to return
#'   classes and memberships. Other types may be provided for some algorithms
#'   (read respective help pages).
#' @param method `"direct"` (default) or `"cv"`. `"direct"` predicts new cases
#'   in `newdata=` if this argument is provided, or the cases in the training
#'   set if not. Take care that not providing `newdata=` means that you just
#'   calculate the **self-consistency** of the classifier but cannot use the
#'   metrics derived from these results for the assessment of its performances.
#'   Either use a different dataset in `newdata=` or use the alternate
#'   cross-validation ("cv") technique. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case. Other
#'   methods may be provided by the various algorithms (check their help pages)
#' @param cv.k k for k-fold cross-validation, cf [ipred::errorest()].
#'   By default, 10.
#' @param cv.strat is the subsampling stratified or not in cross-validation,
#'   cf [ipred::errorest()]. `TRUE` by default.
#'
#' @return an **mlearning** object for [mlearning()]. Methods return their own
#'   results that can be a **mlearning**, **data.frame**, **vector**, etc.
#' @seealso [ml_lda()], [ml_qda()], [ml_naive_bayes()], [ml_nnet()],
#'   [ml_rpart()], [ml_rforest()], [ml_svm()], [confusion()] and [prior()]. Also
#'   [ipred::errorest()] that internally computes the cross-validation
#'   in `cvpredict()`.
#' @export
#'
#' @examples
#' # mlearning() should not be calle directly. Use the mlXXX() functions instead
#' # for instance, for Random Forest, use ml_rforest()/mlRforest()
#' # A typical classification involves several steps:
#' #
#' # 1) Prepare data: split into training set (2/3) and test set (1/3)
#' #    Data cleaning (elimination of unwanted variables), transformation of
#' #    others (scaling, log, ratios, numeric to factor, ...) may be necessary
#' #    here. Apply the same treatments on the training and test sets
#' data("iris", package = "datasets")
#' train <- c(1:34, 51:83, 101:133) # Also random or stratified sampling
#' iris_train <- iris[train, ]
#' iris_test <- iris[-train, ]
#'
#' # 2) Train the classifier, use of the simplified formula class ~ . encouraged
#' #    so, you may have to prepare the train/test sets to keep only relevant
#' #    variables and to possibly transform them before use
#' iris_rf <- ml_rforest(data = iris_train, Species ~ .)
#' iris_rf
#' summary(iris_rf)
#' train(iris_rf)
#' response(iris_rf)
#'
#' # 3) Find optimal values for the parameters of the model
#' #    This is usally done iteratively. Just an example with ntree where a plot
#' #    exists to help finding optimal value
#' plot(iris_rf)
#' # For such a relatively simple case, 50 trees are enough, retrain with it
#' iris_rf <- ml_rforest(data = iris_train, Species ~ ., ntree = 50)
#' summary(iris_rf)
#'
#' # 4) Study the classifier performances. Several metrics and tools exists
#' #    like ROC curves, AUC, etc. Tools provided here are the confusion matrix
#' #    and the metrics that are calculated on it.
#' predict(iris_rf) # Default type is class
#' predict(iris_rf, type = "membership")
#' predict(iris_rf, type = "both")
#' # Confusion matrice and metrics using 10-fols cross-validation
#' iris_rf_conf <- confusion(iris_rf, method = "cv")
#' iris_rf_conf
#' summary(iris_rf_conf)
#' # Note you may want to manipulate priors too, see ?prior
#'
#' # 5) Go back to step #1 and refine the process until you are happy with the
#' #    results. Then, you can use the classifier to predict unknown items.
mlearning <- function(formula, data, method, model.args, call = match.call(),
..., subset, na.action = na.fail) {
  # Our own construction of response vector and terms matrix
  if (missing(model.args))
    model.args <- list(formula  = formula, data = substitute(data),
      subset = substitute(subset))

  # Get data and initial number of cases
  if (missing(data))
    data <- eval.parent(model.args$data)
  data <- as.data.frame(data)
  nobs <- NROW(data)

  # Special case for formula like response ~ . which speeds up calc and
  # uses less memory than model.frame()
  isSimpleFormula <- function(formula) {
    vars <- all.vars(formula)
    (length(vars) == 2 && vars[2] == ".") || # Supervised (response ~ .)
    (length(vars) == 1 && vars[1] == ".")   # Unsupervised (~ .)
  }
  optim <- isSimpleFormula(model.args$formula)
  if (optim) {
    # data do not need to be changed... except for subset or na.action
    if (any(model.args$subset != ""))
      data <- data[eval.parent(model.args$subset), ]
    if (missing(na.action) || any(as.character(na.action) == "")) {
      # Use same rules as model.frame():
      # (1) any na.action attribute of data
      na.action <- attr(data, "na.action")
      # (2) option na.action, or (3) na.fail
      if (is.null(na.action))
        na.action <- getOption("na.action", na.fail)
    }
    # Apply provided na.action
    data <- match.fun(na.action)(data)
    if (is.function(na.action))
      na.action <- substitute(na.action)
    na.action <- as.character(na.action)
    model.terms <- terms(formula, data = data[1, ])
    attr(data, "terms") <- model.terms
  } else {# Use model.frame()
    if (missing(na.action) || any(as.character(na.action) == "")) {
      data <- do.call("model.frame", model.args)
      na.action <- as.character(attr(data, "na.action"))
      if (!length(na.action)) {
        na.action <- "na.pass" # If not provided, either pass, or no NAs!
      } else {
        na.action <- paste("na", class(na.action), sep = ".")
      }
    } else {
      model.args$na.action <- na.action
      data <- do.call("model.frame", model.args)
      if (is.function(na.action))
        na.action <- substitute(na.action)
      na.action <- as.character(na.action)
    }
    model.terms <- attr(data, "terms")
  }
  # Final number of observations
  nobs[2] <- NROW(data)
  names(nobs) <- c("initial", "final")

  # Construct the matrix of numeric predictors and the response
  term.labels <- attr(model.terms, "term.labels")
  response.pos <- attr(model.terms, "response")
  if (!response.pos[1]) {
    response.label <- NULL
    train <- data
    response <- NULL
    lev <- NULL
    type <- "unsupervised"
  } else {# Supervised classification or regression
    response.label <- deparse(attr(model.terms, "variables")
      [[response.pos + 1]])
    response <- data[[response.label]]
    if (is.factor(response)) {
      lev <- levels(response)
      response <- droplevels(response)
      type <- "classification"
    } else {
      if (!is.numeric(response))
        stop("response variable must be factor or numeric")
      lev <- NULL
      type <- "regression"
    }
    train <- data[, term.labels]
  }

  # Calculate weights
  w <- model.weights(data)
  if (length(w) == 0L)
    w <- rep(1, nrow(train))

  # Pass special arguments to the default method
  args <- list()
  args$formula <- formula
  args$levels <- lev
  args$n <- nobs
  args$weights <- w
  args$optim <- optim
  args$type <- type
  args$na.action <- substitute(na.action)
  args$mlearning.call <- call
  args$method <- method

  # Construct the mlearning object
  match.fun(method)(train = train, response = response, .args. = args, ...)
}

#' @rdname mlearning
#' @export
#' @method print mlearning
print.mlearning <- function(x, ...) {
  cat("A mlearning object of class ", class(x)[1], " (",
    attr(x, "algorithm"), "):\n", sep = "")
  type <- attr(x, "type")
  switch(type,
    regression = cat("[regression variant]\n"),
    unsupervised = cat("[unsupervised classification variant]\n"))
  cat("Call: ", deparse(attr(x, "mlearning.call")), "\n", sep = "")

  if (type[1] == "classification") {
    # Number of cases used
    n <- attr(x, "n")
    if (any(n["final"] < n["initial"])) {
      msg <- paste("Trained using", n["final"], "out of",
        n["initial"], "cases:")
    } else {
      msg <- paste("Trained using", n["final"], "cases:")
    }

    # Categories
    classes <- attr(x, "response")
    levUsed <- levels(classes)
    levIni <- levels(x)
    if (length(levUsed) < length(levIni)) {
      cat("Levels with no cases in the training set that were eliminated:\n")
      print(levIni[!levIni %in% levUsed])
    }

    # Number of cases per used categories
    print(table(classes, dnn = msg))
  }
  invisible(x)
}

#' @rdname mlearning
#' @export
#' @method summary mlearning
summary.mlearning <- function(object, ...) {
  train <- attr(object, "train")
  response <- attr(object, "response")
  mlearning.class <- class(object)[1]
  class(object) <- class(object)[-(1:2)]
  ## Summary is sometimes implemented as print() for some machine
  ## learning algorithms... this is is 'summary' attribute
  sumfun <- attr(object, "summary")
  if (length(sumfun)) {
    res <- match.fun(sumfun)(object, ...)
  } else {
    res <- object
  }
  class(res) <- c("summary.mlearning", class(res))
  attr(res, "mlearning.class") <- mlearning.class
  attr(res, "algorithm") <- attr(object, "algorithm")
  attr(res, "type") <- attr(object, "type")
  attr(res, "mlearning.call") <- attr(object, "mlearning.call")
  res
}

#' @rdname mlearning
#' @export
#' @method print summary.mlearning
print.summary.mlearning <- function(x, ...) {
  cat("A mlearning object of class ", attr(x, "mlearning.class"), " (",
    attr(x, "algorithm"), "):\n", sep = "")
  type <- attr(x, "type")
  switch(type,
    regression = cat("[regression variant]\n"),
    unsupervised = cat("[unsupervised classification variant]\n"))
  cat("Initial call: ", deparse(attr(x, "mlearning.call")), "\n", sep = "")
  X <- x
  class(X) <- class(x)[-1]
  print(X)
  invisible(x)
}

#' @rdname mlearning
#' @export
#' @method plot mlearning
plot.mlearning <- function(x, y, ...) {
  train <- attr(x, "train")
  response <- attr(x, "response")
  class(x) <- class(x)[-(1:2)]
  plot(x, ...)
}

.membership <- function(x, levels, scale = TRUE) {
  # Make sure x is a matrix of numerics
  x <- as.matrix(x)
  if (!is.numeric(x))
    stop("'x' must be numeric")

  # Make sure all columns are named with names in levels
  nms <- colnames(x)
  if (!length(nms))
    stop("missing column names in 'x'")
  if (any(!nms %in% levels))
    stop("One or more column in 'x' not in 'levels'")

  # Add columns of zeros for inexistant levels
  toAdd <- levels[!levels %in% nms]
  if (length(toAdd)) {
    xAdd <- matrix(0, nrow = NROW(x), ncol = length(toAdd))
    colnames(xAdd) <- toAdd
    x <- cbind(x, xAdd)
  }

  # Make sure columns are in the same order as levels
  x <- x[, levels]

  # Possibly scale to one, row-wise
  if (isTRUE(as.logical(scale)))
    x <- x / apply(x, 1, sum)

  x
}

.expandFactor <- function(f, n, ndrop)
{
  if (!length(ndrop) || !inherits(ndrop, "exclude"))
    return(f)
  res <- factor(rep(NA, n), levels = levels(f))
  res[-ndrop] <- f
  res
}

.expandMatrix <- function(m, n, ndrop) {
  if (!length(ndrop) || !inherits(ndrop, "exclude"))
    return(m)
  res <- matrix(NA, nrow = n, ncol = ncol(m))
  res[-ndrop, ] <- m
  res
}

#' @rdname mlearning
#' @export
#' @method predict mlearning
predict.mlearning <- function(object, newdata,
type = c("class", "membership", "both"), method = c("direct", "cv"),
na.action = na.exclude, ...) {
  # Not usable for unsupervised type
  if (attr(object, "type")[1] == "unsupervised")
    stop("no predict() method for unsupervised version")

  # If method == "cv", delegate to cvpredict()
  if (as.character(method)[1] == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, ...))
  }

  # Recalculate newdata according to formula...
  if (missing(newdata)) {# Use train
    newdata <- attr(object, "train")
  } else if (attr(object, "optim")[1]) {# Use optimized approach
    # Just keep vars similar as in train
    vars <- names(attr(object, "train"))
    if (!all(vars %in% names(newdata)))
      stop("one or more missing variables in newdata")
    newdata <- newdata[, vars]
  } else {# Use model.frame
    # but eliminate dependent variable, not required
    # (second item in the formula)
    newdata <- model.frame(formula = attr(object, "formula")[-2],
      data = newdata, na.action = na.pass)[, names(attr(object, "train"))]
  }
  # Do we need only numerical predictors
  if (attr(object, "numeric.only")[1])
    newdata <- sapply(as.data.frame(newdata), as.numeric)

  # Determine how many data and perform na.action
  n <- NROW(newdata)
  newdata <- match.fun(na.action)(newdata)
  ndrop <- attr(newdata, "na.action")

  # Delegate to the original predict() method
  class(object) <- class(object)[-(1:2)]
  if (attr(object, "type")[1] == "regression")
    return(predict(object, newdata = newdata, ...))

  # Otherwise, this is a supervised classification
  type <- as.character(type)[1]
  ## Special case for both
  if (type == "both")
    type <- c("class", "membership")
  # Check that type is supported and look for corresponding type name
  # in original predict() method
  pred.type <- attr(object, "pred.type")
  if (!all(type %in% names(pred.type)))
    stop("unsupported predict type")

  if (length(type) == 2) {
    # Special case where we predict both class and membership
    classes <- predict(object, newdata = newdata,
      type = pred.type["class"], ...)
    members <- predict(object, newdata = newdata,
      type = pred.type["membership"], ...)
    # Create a list with both res
    levels <- levels(object$predicted)
    return(list(class = .expandFactor(factor(as.character(classes),
      levels = levels), n, ndrop),
      membership = .expandMatrix(.membership(members, levels = levels),
      n, ndrop)))
  } else {
    res <- predict(object, newdata = newdata, type = pred.type[type], ...)
  }

  # Rework result according to initial levels (before drop of empty ones)
  res <- switch(type,
    class = .expandFactor(factor(as.character(res),
      levels = levels(object$predicted)), n, ndrop),
    membership = .expandMatrix(.membership(res,
      levels = levels(object$predicted)), n, ndrop),
    switch(class(res)[1],
      factor = .expandFactor(res, n, ndrop),
      matrix = .expandMatrix(res, n, ndrop),
      res))

  res
}

#' @rdname mlearning
#' @export
cvpredict <- function(object, ...)
  UseMethod("cvpredict")

#' @rdname mlearning
#' @export
#' @method cvpredict mlearning
cvpredict.mlearning <- function(object, type = c("class", "membership", "both"),
cv.k = 10, cv.strat = TRUE, ...) {
  type <- switch(attr(object, "type"),
    regression = "class", # Another way to ignore 'type' for regressions
    classification = as.character(type)[1],
    stop("works only for classification or regression mlearning objects"))

  if (type[1] == "class") {
    predictions <- TRUE
    getmodels <- FALSE
  } else if (type[1] == "membership") {
    predictions <- FALSE
    getmodels <- TRUE
  } else if (type[1] == "both") {
    predictions <- TRUE
    getmodels <- TRUE
  } else stop("type must be 'class', 'membership' or 'both'")

  # Create data, using numbers are rownames
  data <- data.frame(.response. = response(object), train(object))
  rn <- rownames(data)
  if (is.null(rn))
    rn <- 1:NROW(data)
  rownames(data) <- 1:NROW(data)

  # The predict() method with ... arguments added to the call
  constructPredict <- function(...) {
    fun <- function(object, newdata) return()
    body(fun) <- as.call(c(list(substitute(predict),
      object = substitute(object), newdata = substitute(newdata)), list(...)))
    fun
  }
  Predict <- constructPredict(...)

  # Perform cross-validation for prediction
  args <- attr(object, "args")
  if (!is.list(args)) args <- list()
  args$formula <- substitute(.response. ~ .)
  args$data <- substitute(data)
  args$model <- substitute(mlearning)
  args$method <- attr(object, "method")
  args$predict <- substitute(Predict)
  args$estimator <- "cv"
  args$est.para <- ipred::control.errorest(predictions = predictions,
    getmodels = getmodels, k = cv.k, strat = cv.strat)
  est <- do.call(errorest, args)

  # Only class
  if (type[1] == "class") {
    res <- est$predictions
  } else {
    # Need to calculate membership
    predCV <- function(x, object, ...) {
      Train <- train(object)
      rownames(Train) <- 1:NROW(Train)
      suppressWarnings(predict(x, newdata =
        Train[-as.numeric(rownames(train(x))), ], ...))
    }

    # Apply predict on all model and collect results together
    membership <- lapply(est$models, predCV, object = object,
      type = "membership", na.action = na.exclude, ...)

    # Concatenate results
    membership <- do.call(rbind, membership)

    # Sort in correct order and replace initial rownames
    ord <- as.numeric(rownames(membership))
    # Sometimes, errorest() duplicates one or two items in two models
    # (rounding errors?) => eliminate them here
    # Note: commented out, because it does not work with mlSvm()!
    notDup <- !duplicated(ord)
    membership <- membership[notDup, ]
    ord <- ord[notDup]

    # Restore order of the items
    rownames(membership) <- rn[ord]
    pos <- order(ord)
    membership <- membership[pos, ]

    if (type[1] == "membership") {
      res <- membership
    } else {# Need both class and membership
      # Because we don't know who is who in est$predictions in case of
      # duplicated items in est$models, we prefer to recalculate classes
      classes <- unlist(lapply(est$models, predCV, object = object,
        type = "class", na.action = na.exclude, ...))
      classes <- classes[notDup]
      classes <- classes[pos]

      # Check that both classes levels are the same!
      if (any(levels(classes) != levels(est$predictions)))
        warning("cross-validated classes do not match")

      res <- list(class = classes, membership = membership)
    }
  }

  # Add est object as "method" attribute, without predictions or models
  est$name <- "cross-validation"
  est$predictions <- NULL
  est$models <- NULL
  est$call <- match.call()
  est$strat <- cv.strat
  attr(res, "method") <- est

  res
}
