#' Supervised classification and regression using support vector machine
#'
#' @description
#' Unified (formula-based) interface version of the support vector machine
#' algorithm provided by [e1071::svm()].
#'
#' @param formula a formula with left term being the factor variable to predict
#' (for supervised classification), a vector of numbers (for regression) or
#' nothing (for unsupervised classification) and the right term with the list
#' of independent, predictive variables, separated with a plus sign. If the
#' data frame provided contains only the dependent and independent variables,
#' one can use the `class ~ .` short version (that one is strongly encouraged).
#' Variables with minus sign are eliminated. Calculations on variables are
#' possible according to usual formula convention (possibly protected by using
#' `I()`).
#' @param data a data.frame to use as a training set.
#' @param train a matrix or data frame with predictors.
#' @param response a vector of factor (classification) or numeric (regression).
#' @param scale are the variables scaled (so that mean = 0 and standard
#'   deviation = 1)? `TRUE` by default. If a vector is provided, it is applied
#'   to variables with recycling.
#' @param type For [ml_svm()]/[mlSvm()], the type of classification or
#'   regression machine to use. The default value of `NULL` uses
#'   `"C-classification"` if response variable is factor and  `eps-regression`
#'   if it is numeric. It can also be `"nu-classification"` or
#'   `"nu-regression"`. The "C" and "nu" versions are basically the same but
#'   with a different parameterisation. The range of C is from zero to infinity,
#'   while the range for nu is from zero to one. A fifth option is
#'   `"one_classification"` that is specific to novelty detection (find the
#'   items that are different from the rest).
#'   For [predict()], the type of prediction to return. `"class"` by default,
#'   the predicted classes. Other options are `"membership"` the membership
#'   (number between 0 and 1) to the different classes, or  `"both"` to return
#'   classes and memberships.
#' @param kernel the kernel used by svm, see [e1071::svm()] for further
#'   explanations. Can be `"radial"`, `"linear"`, `"polynomial"` or `"sigmoid"`.
#' @param classwt priors of the classes. Need not add up to one.
#' @param ... further arguments passed to the classification or regression
#'   method. See [e1071::svm()].
#' @param subset index vector with the cases to define the training set in use
#'   (this argument must be named, if provided).
#' @param na.action function to specify the action to be taken if `NA`s are
#'   found. For [ml_svm()] `na.fail` is used by default. The calculation is
#'   stopped if there is any `NA` in the data. Another option is `na.omit`,
#'   where cases with missing values on any required variable are dropped (this
#'   argument must be named, if provided). For the `predict()` method, the
#'   default, and most suitable option, is `na.exclude`. In that case, rows with
#'   `NA`s in `newdata=` are excluded from prediction, but reinjected in the
#'   final results so that the number of items is still the same (and in the
#'   same order as `newdata=`).
#' @param object an **mlSvm** object
#' @param newdata a new dataset with same conformation as the training set (same
#'   variables, except may by the class for classification or dependent variable
#'   for regression). Usually a test set, or a new dataset to be predicted.
#' @param method `"direct"` (default) or `"cv"`. `"direct"` predicts new cases in
#'   `newdata=` if this argument is provided, or the cases in the training set
#'   if not. Take care that not providing `newdata=` means that you just
#'   calculate the **self-consistency** of the classifier but cannot use the
#'   metrics derived from these results for the assessment of its performances.
#'   Either use a different data set in `newdata=` or use the alternate
#'   cross-validation ("cv") technique. If you specify `method = "cv"` then
#'   [cvpredict()] is used and you cannot provide `newdata=` in that case.
#'
#' @return [ml_svm()]/[mlSvm()] creates an **mlSvm**, **mlearning** object
#'   containing the classifier and a lot of additional metadata used by the
#'   functions and methods you can apply to it like [predict()] or
#'   [cvpredict()]. In case you want to program new functions or extract
#'   specific components, inspect the "unclassed" object using [unclass()].
#' @seealso [mlearning()], [cvpredict()], [confusion()], also [e1071::svm()]
#'   that actually does the calculation.
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
#' iris_svm <- ml_svm(data = iris_train, Species ~ .)
#' summary(iris_svm)
#' predict(iris_svm) # Default type is class
#' predict(iris_svm, type = "membership")
#' predict(iris_svm, type = "both")
#' # Self-consistency, do not use for assessing classifier performances!
#' confusion(iris_svm)
#' # Use an independent test set instead
#' confusion(predict(iris_svm, newdata = iris_test), iris_test$Species)
#'
#' # Another dataset
#' data("HouseVotes84", package = "mlbench")
#' house_svm <- ml_svm(data = HouseVotes84, Class ~ ., na.action = na.omit)
#' summary(house_svm)
#' # Cross-validated confusion matrix
#' confusion(cvpredict(house_svm), na.omit(HouseVotes84)$Class)
#'
#' # Regression using support vector machine
#' data(airquality, package = "datasets")
#' ozone_svm <- ml_svm(data = airquality, Ozone ~ ., na.action = na.omit)
#' summary(ozone_svm)
#' plot(na.omit(airquality)$Ozone, predict(ozone_svm))
#' abline(a = 0, b = 1)
mlSvm <- function(train, ...)
  UseMethod("mlSvm")

#' @rdname mlSvm
#' @export
ml_svm <- mlSvm

#' @rdname mlSvm
#' @export
#' @method mlSvm formula
mlSvm.formula <- function(formula, data, scale = TRUE, type = NULL,
kernel = "radial", classwt = NULL, ..., subset, na.action) {
  mlearning(formula, data = data, method = "mlSvm", model.args =
    list(formula  = formula, data = substitute(data),
      subset = substitute(subset)), call = match.call(),
    scale = scale, type = type, kernel = kernel, classwt = classwt, ...,
    subset = subset, na.action = substitute(na.action))
}

#' @rdname mlSvm
#' @export
#' @method mlSvm default
mlSvm.default <- function(train, response, scale = TRUE, type = NULL,
kernel = "radial", classwt = NULL, ...) {
  dots <- list(...)
  .args. <- dots$.args.
  dots$.args. <- NULL
  if (!length(.args.)) {
    if (is.factor(response)) {
      Type <- "classification"
    } else {
      Type <- "regression"
    }
    .args. <- list(levels = levels(response),
      n = c(intial = NROW(train), final = NROW(train)),
      type = Type, na.action = "na.pass",
      mlearning.call = match.call(), method = "mlSvm")
  }
  dots$scale <- scale
  dots$type <- type
  dots$kernel <- kernel
  dots$class.weigths <- classwt
  #dots$probability <- TRUE

  # Return a mlearning object
  structure(e1071::svm(x = sapply(train, as.numeric), y = response,
    scale = scale, type = type, kernel = kernel, class.weights = classwt,
    probability = TRUE, ...), formula = .args.$formula, train = train,
    response = response, levels = .args.$levels, n = .args.$n, args = dots,
    optim = .args.$optim, numeric.only = TRUE, type = .args.$type,
    pred.type = c(class = "class", membership = "raw"),
    summary = "summary", na.action = .args.$na.action,
    mlearning.call = .args.$mlearning.call, method = .args.$method,
    algorithm = "support vector machine",
    class = c("mlSvm", "mlearning", "svm"))
}

#' @rdname mlSvm
#' @export
#' @method predict mlSvm
predict.mlSvm <- function(object, newdata,
type = c("class", "membership", "both"), method = c("direct", "cv"),
na.action = na.exclude, ...) {
  if (!inherits(object, "mlSvm"))
    stop("'object' must be a 'mlSvm' object")

  # If method == "cv", delegate to cvpredict()
  method <- as.character(method)[1]
  if (method == "cv") {
    if (!missing(newdata))
      stop("cannot handle new data with method = 'cv'")
    return(cvpredict(object = object, type = type, ...))
  }

  # Recalculate newdata according to formula...
  if (missing(newdata)) { # Use train
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

  # Delegate to the e1071 predict.svm method
  if (as.character(type)[1] == "class") proba <- FALSE else proba <- TRUE
  class(object) <- class(object)[-(1:2)]
  if (attr(object, "type")[1] == "regression")
    return(predict(object, newdata = newdata, ...))

  # This is for classification
  res <- predict(object, newdata = newdata,
    probability = proba, ...)
  proba <- attr(res, "probabilities")

  # Rework results according to what we want
  switch(as.character(type)[1],
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
