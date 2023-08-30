#' Construct and analyze confusion matrices
#'
#' @description
#' Confusion matrices compare two classifications (usually one done
#' automatically using a machine learning algorithm versus the true
#' classification done by a specialist... but one can also compare two automatic
#' or two manual classifications against each other).
#'
#' @param x an object with a `confusion()` method implemented.
#' @param y another object, from which to extract the second classification, or
#'   `NULL` if not used.
#' @param vars the variables of interest in the first and second classification
#'   in the case the objects are lists or data frames. Otherwise, this argument
#'   is ignored and `x` and `y` must be factors with same length and same levels.
#' @param labels labels to use for the two classifications. By default, they are
#'   the same as `vars`, or the one in the confusion matrix.
#' @param merge.by a character string with the name of variables to use to merge
#'   the two data frames, or `NULL`.
#' @param useNA do we keep `NA`s as a separate category? The default `"ifany"`
#'   creates this category only if there are missing values. Other possibilities
#'   are `"no"`, or `"always"`.
#' @param prior class frequencies to use for first classifier that is tabulated
#'   in the rows of the confusion matrix. For its value, see here under, the
#'   `value=` argument.
#' @param ... further arguments passed to the method.
#'
#' @return A confusion matrix in a **confusion** object.
#' @export
#' @seealso [mlearning()], [plot.confusion()], [prior()]
#' @examples
#' data("Glass", package = "mlbench")
#' # Use a little bit more informative labels for Type
#' Glass$Type <- as.factor(paste("Glass", Glass$Type))
#'
#' # Use learning vector quantization to classify the glass types
#' # (using default parameters)
#' summary(glass_lvq <- ml_lvq(Type ~ ., data = Glass))
#'
#' # Calculate cross-validated confusion matrix
#' (glass_conf <- confusion(cvpredict(glass_lvq), Glass$Type))
#' # Raw confusion matrix: no sort and no margins
#' print(glass_conf, sums = FALSE, sort = FALSE)
#'
#' summary(glass_conf)
#' summary(glass_conf, type = "Fscore")
confusion <- function(x, ...)
  UseMethod("confusion")

# TODO: implement weights
.confusion <- function(classes, labels, useNA, prior, ...) {
  # useNA can be "no", "always" or "ifany", but with the later value
  # one takes the risk to get non square matrix if there are NAs in only
  # one vector of classes => change to "no" or "always", depending if there
  # are missing data or not
  if (useNA == "ifany")
    if (any(is.na(classes))) useNA <- "always" else useNA <- "no"
  res <- table(classes, dnn = labels, useNA = useNA)
  total <- sum(res)
  truePos <- sum(diag(res))
  row.freqs <- rowSums(res)

  # Additional data as attributes
  attr(res, "row.freqs") <- row.freqs
  attr(res, "col.freqs") <- colSums(res)
  attr(res, "levels") <- levels(classes[1, ]) # These are *initial* levels!
  # Final levels may differ if there are empty levels, or NAs!
  attr(res, "prior") <- row.freqs # Initial prior are row.freqs
  attr(res, "stats") <- c(total = total, truepos = truePos,
    error = 1 - (truePos / total))

  # This is a confusion object, inheriting from table
  class(res) <- c("confusion", "table")

  # Do we rescale the confusion matrix?
  if (!missing(prior)) prior(res) <- prior

  res
}

#' @rdname confusion
#' @export
#' @method confusion default
confusion.default <- function(x, y = NULL, vars = c("Actual", "Predicted"),
labels = vars, merge.by = "Id", useNA = "ifany", prior, ...) {
  # If the object is already a 'confusion' object, return it
  if (inherits(x, "confusion")) {
    if (!missing(y))
    warning("you cannot provide 'y' when 'x' is a 'confusion' object")
    # Possibly rescale it
    if (!missing(prior)) prior(x) <- prior
    x
  }

  # Idem if there is a 'confusion' attribute and no y
  conf <- attr(x, "confusion")
  if (!is.null(conf) && missing(y)) {
    # Possibly reweight it
    if (!missing(prior)) prior(conf) <- prior
    conf
  }

  # Reworks and check arguments
  vars <- as.character(vars)
  if (length(vars) != 2)
    stop("You must provide exactly 2 strings for 'vars'")
  merge.by <- as.character(merge.by)

  # There are three possibilities:
  # 1) A single data frame => use vars
  if (missing(y)) {
    # Special case of a data frame or list of two factors: keep as it is
    if (is.list(x) && length(x) == 2 && is.null(vars)) {
      clCompa <- as.data.frame(x)
      if (missing(labels)) labels <- names(clCompa)
    } else {
      x <- as.data.frame(x)
      # Check that vars exist
      if (is.null(names(x)) || !all(vars %in% names(x)))
        stop("'vars' are not among column names of 'x'")
      # Check that levels of two vars do match
      lev1 <- levels(x[[vars[1]]])
      lev2 <- levels(x[[vars[2]]])
      if (!all(lev1 == lev2)) {
        # If difference is only in the order of both levels, reorder #2
        if (!all(sort(lev1) == sort(lev2))) {
          stop("levels of the two variables in 'x' do not match")
        } else x[[vars[2]]] <- factor(as.character(x[[vars[2]]]),
          levels = lev1)
      }
      clCompa <- data.frame(class1 = x[[vars[1]]], class2 = x[[vars[2]]])
    }
  } else {# y is provided
    # 2) Two vectors of factors (must have same length/same levels)
    if (is.factor(x) && is.factor(y)) {
      # Check length match
      if (length(x) != length(x))
        stop("lengths of 'x' and 'y' are not the same")
      # Check levels match
      lev1 <- levels(x)
      lev2 <- levels(y)
      if (!all(lev1  == lev2)) {
        # If difference is only in the order of both levels, reorder #2
        if (!all(sort(lev1)  == sort(lev1))) {
          stop("'x' and 'y' levels do not match")
        } else {
          y <- factor(as.character(y), levels = lev1)
        }
      }
      clCompa <- data.frame(class1 = y, class2 = x)
    } else {
      # 3) Two data frames => merge first, then use vars
      # Check that vars exists
      if (is.null(names(x)) || !(vars[1] %in% names(x)))
        stop("first item of 'vars' is not among names of 'x'")
      if (is.null(names(y)) || !(vars[2] %in% names(y)))
        stop("second item of 'vars' is not among names of 'y'")
      # Check that levels of two vars do match
      lev1 <- levels(x[[vars[1]]])
      lev2 <- levels(y[[vars[2]]])
      if (!all(lev1  == lev2)) {
        # If difference is only in the order of both levels, reorder #2
        if (!all(sort(lev1)  == sort(lev2))) {
          stop("levels of the variables in 'x' and 'y' do not match")
        } else {
          x[[vars[2]]] <- factor(as.character(x[[vars[2]]]), levels = lev1)
        }
      }
      # Merge data according to merge.by
      clCompa <- merge(y[, c(vars[2], merge.by)],
        x[, c(vars[1], merge.by)], by = merge.by)
      nc <- ncol(clCompa)
      clCompa <- clCompa[, c(nc - 1, nc)]
      # Are there common objects left?
      if (!nrow(clCompa)) stop("no common objects between 'x' and 'y'")
    }
  }

  # Construct the confusion object
  if (missing(prior)) {
    .confusion(classes = clCompa, labels = labels, useNA = useNA, ...)
  } else {
    .confusion(classes = clCompa, labels = labels, useNA = useNA,
      prior = prior, ...)
  }
}

#' @rdname confusion
#' @export
#' @method confusion mlearning
confusion.mlearning <- function(x, y = response(x),
labels = c("Actual", "Predicted"), useNA = "ifany", prior, ...) {
  # Check labels
  labels <- as.character(labels)
  if (length(labels) != 2)
    stop("You must provide exactly 2 character strings for 'labels'")

  # Extract class2 by using predict on the mlearning object
  class2 <- predict(x, ...)

  # Check that both variables are of same length and same levels
  if (length(y) != length(class2))
    stop("lengths of 'x' and 'y' are not the same")
  lev1 <- levels(y)
  lev2 <- levels(class2)
  if (!all(lev1  == lev2)) {
    # If difference is only in the order of both levels, reorder #2
    if (!all(sort(lev1)  == sort(lev2))) {
      stop("levels of 'x' and 'y' do not match")
    } else {
      class2 <- factor(as.character(class2), levels = lev1)
    }
  }

  # Construct the confusion object
  if (missing(prior)) {
    .confusion(data.frame(class1 = y, class2 = class2),
      labels = labels, useNA = useNA, ...)
  } else {
    .confusion(data.frame(class1 = y, class2 = class2),
      labels = labels, useNA = useNA, prior = prior, ...)
  }
}

#' @rdname confusion
#' @export
#' @method print confusion
#' @param sums is the confusion matrix printed with rows and columns sums?
#' @param error.col is a column with class error for first classifier added
#'   (equivalent to false negative rate of FNR)?
#' @param digits the number of digits after the decimal point to print in the
#'   confusion matrix. The default or zero leads to most compact presentation
#'   and is suitable for frequencies, but not for relative frequencies.
#' @param sort are rows and columns of the confusion matrix sorted so that
#'   classes with larger confusion are closer together? Sorting is done
#'   using a hierarchical clustering with [hclust()]. The clustering method
#'   is `"ward.D2"` by default, but see the [hclust()] help for other options).
#'   If `FALSE` or `NULL`, no sorting is done.
print.confusion <- function(x, sums = TRUE, error.col = sums, digits = 0,
sort = "ward.D2", ...) {
  # General stats on the confusion matrix
  Stats <- attr(x, "stats")
  Error <- round(Stats["error"] * 100, 1)
  cat(Stats["total"], " items classified with ", Stats["truepos"],
    " true positives (error rate = ", Error, "%)\n", sep = "")
  row.freqs <- attr(x, "row.freqs")
  if (!all(attr(x, "prior") == row.freqs)) {
    cat("with initial row frequencies:\n")
    print(row.freqs)
    cat("Rescaled to:\n")
  }

  # Print the confusion matrix itself
  X <- x
  class(X) <- "table"

  n <- ncol(X)

  # Do we sort items?
  if (length(sort) && any(!is.na(sort)) && any(sort != FALSE) &&
    any(sort != "")) {
    # Grouping of items
    confuSim <- X + t(X)
    confuSim <- 1 - (confuSim / sum(confuSim) * 2)
    confuDist <- structure(confuSim[lower.tri(confuSim)], Size = n,
      Diag = FALSE, Upper = FALSE, method = "confusion", call = "",
      class = "dist")
    order <- hclust(confuDist, method = sort)$order
    X <- X[order, order]
  }

  # Change row and column names to a more compact representation
  nbrs <- formatC(1:ncol(X), digits = 1, flag = "0")
  colnames(X) <- nbrs
  rownames(X) <- paste(nbrs, rownames(X))

  # Add sums?
  if (isTRUE(as.logical(sums))) {
    # Calculate error (%)
    ErrorTot <- (1 - (sum(diag(x)) / sum(x))) * 100
    Errors <- as.integer(round(c((1 - diag(X) / apply(X, 1, sum)) * 100,
      ErrorTot), 0))
    # ... and add row and column sums
    X <- addmargins(X, FUN = list(`(sum)` = sum), quiet = TRUE)
  } else {
    Errors <- as.integer(round((1 - diag(X) / apply(X, 1, sum)) * 100, 0))
  }

  # Add class errors?
  if (isTRUE(as.logical(error.col))) {
    X <- as.table(cbind(X, `(FNR%)` = Errors))
    dn <- dimnames(X)
    names(dn) <- names(dimnames(x))
    dimnames(X) <- dn
  }
  print(round(X, digits))

  # Return the original object invisibly
  invisible(x)
}

#' @rdname confusion
#' @export
#' @method summary confusion
#' @param object a **confusion** object
#' @param type either `"all"` (by default), or considering `TP` is the true
#' positives, `FP` is the false positives, `TN` is the true negatives and `FN`
#' is the false negatives, one can also specify: `"Fscore"` (F-score = F-measure
#' = F1 score = harmonic mean of Precision and recall), `"Recall"`
#' (TP / (TP + FN) = 1 - FNR), `"Precision"` (TP / (TP + FP) = 1 - FDR),
#' `"Specificity"` (TN / (TN + FP) = 1 - FPR), `"NPV"` (Negative predicted value
#' = TN / (TN + FN) = 1 - FOR), `"FPR"` (False positive rate = 1 - Specificity
#' = FP / (FP + TN)), `"FNR"` (False negative rate = 1 - Recall = FN / (TP + FN)),
#' `"FDR"` (False Discovery Rate = 1 - Precision = FP / (TP + FP)), `"FOR"`
#' (False omission rate = 1 - NPV = FN / (FN + TN)), `"LRPT"` (Likelihood Ratio
#' for Positive Tests = Recall / FPR = Recall / (1 - Specificity)), `"LRNT"`
#' Likelihood Ratio for Negative Tests = FNR / Specificity = (1 - Recall) /
#' Specificity, `"LRPS"` (Likelihood Ratio for Positive Subjects = Precision /
#' FOR = Precision / (1 - NPV)), `"LRNS"` (Likelihood Ratio Negative Subjects =
#' FDR / NPV = (1 - Precision) / (1 - FOR)), `"BalAcc"` (Balanced accuracy
#' = (Sensitivity + Specificity) / 2), `"MCC"` (Matthews correlation coefficient),
#'   `"Chisq"` (Chisq metric), or `"Bray"` (Bray-Curtis metric)
#' @param sort.by the statistics to use to sort the table (by default, Fmeasure,
#' the F1 score for each class = 2 * recall * precision / (recall + precision)).
#' @param decreasing do we sort in increasing or decreasing order?
summary.confusion <- function(object, type = "all", sort.by = "Fscore",
decreasing = TRUE, ...) {
  # Check objects
  if (!inherits(object, "confusion"))
    stop("'object' must be a 'confusion' object")

  # General parameters
  ## Number of groups
  Ngp <- nrow(object)

  # Total : TP + TN + FP + FN
  Tot <- sum(object)

  # TP : True positive item : All items on diagonal
  TP <- diag(object)

  # TP + TN : sum of diagonal = All correct identification
  TP_TN <- sum(TP)

  # TP + FP : sum of columns : Automatic classification
  TP_FP <- colSums(object)

  # TP + FN : sum of rows : Manual classification
  TP_FN <- rowSums(object)

  # FP : False positive items
  FP <- TP_FP - TP

  # FN : False negative item
  FN <- TP_FN - TP

  # TN : True Negative = Total - TP - FP - FN
  TN <- rep(Tot, Ngp) - TP - FP - FN

  # The 8 basic ratios
  # Recall = TP / (TP + FN) = 1 - FNR
   Recall <- TP / (TP_FN)

  # Specificity = TN / (TN + FP) = 1 - FPR
  Specificity <- TN / (TN + FP)

  # Precision = TP / (TP + FP) = 1 - FDR
  Precision <- TP / (TP_FP)

  # NPV : Negative predicted value = TN / (TN + FN) = 1 - FOR
  NPV <- TN / (TN + FN)

  # FPR : False positive rate = 1 - Specificity = FP / (FP + TN)
  FPR <- FP / (FP + TN) #1 - Specificity

  # FNR : False negative rate = 1 - Recall = FN / (TP + FN)
  FNR <- FN / (TP + FN) #1 - Recall

  # FDR : False Discovery Rate = 1 - Precision = FP / (TP + FP)
  FDR <- FP / (TP_FP) #1 - Precision

  # FOR : False omission rate = 1 - NPV = FN / (FN + TN)
  FOR <- FN / (FN + TN) #1 - NPV

  # The 4 ratios of ratios
  # LRPT = Likelihood Ratio for Positive Tests = Recall / FPR = Recall /
  # (1 - Specificity)
  LRPT <- Recall / (FPR)

  # LRNT = Likelihood Ratio for Negative Tests = FNR / Specificity =
  # (1 - Recall) / Specificity
  LRNT <- FNR / (Specificity)

  # LRPS : Likelihood Ratio for Positive Subjects = Precision / FOR =
  # Precision / (1 - NPV)
  LRPS <- Precision / (FOR)

  # LRNS : Likelihood Ratio Negative Subjects = FDR / NPV = (1 - Precision) /
  # (1 - FOR)
  LRNS <- FDR / (NPV)

  # Additional statistics
  # F-score = F-measure = F1 score = Harmonic mean of Precision and recall
  Fscore <- 2 * ((Precision * Recall) / (Precision + Recall))
  # F-score is also TP/(TP + (FP + FN) / 2). As such, if TP is null but
  # at least one of FP or FN is not null, F-score = 0. In this case,
  # as both Recall and Precision equal zero, we got NaN => do the correction!
  Fscore[is.nan(Fscore)] <- 0

  # Balanced accuracy = (Sensitivity + Specificity) / 2
  BalAcc <- (Recall + Specificity) / 2

  # MCC : Matthews correlation coefficient
  Sum1 <- TP + FP
  Sum2 <- TP + FN
  Sum3 <- TN + FP
  Sum4 <- TN + FN
  Denominator <- sqrt(Sum1 * Sum2 * Sum3 * Sum4)
  ZeroIdx <- Sum1 == 0 | Sum2 == 0 | Sum3 == 0 | Sum4 == 0
  if (any(!is.na(ZeroIdx)) && any(ZeroIdx))
    Denominator[ZeroIdx] <- 1
  MCC <- ((TP * TN) - (FP * FN)) / Denominator

  # Chisq : Significance
  Chisq <- (((TP * TN) - (FP * FN))^2 * (TP + TN + FP + FN)) /
    ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

  # Automatic classification - Manual classification
  Auto_Manu <- TP_FP - TP_FN

  # Bray-Curtis dissimilarity index
  Bray <- abs(Auto_Manu) / (sum(TP_FP) + sum(TP_FN))

  # General statistics
  # Error = 1 - Accuracy = 1 - ((TP + TN) / (TP + TN + FP + FN))
  Error <- 1 - (TP_TN / Tot)
  # Micro and macro-averaged F-score
  meanRecall <- sum(Recall, na.rm = TRUE) / Ngp
  meanPrecision <- sum(Precision, na.rm = TRUE) / Ngp
  Fmicro <- 2 * meanRecall * meanPrecision / (meanRecall + meanPrecision)
  Fmacro <- sum(Fscore, na.rm = TRUE) / Ngp

  # Take care to avoid missing data for data frame rownames!
  nms <- names(Fscore)
  nms[is.na(nms)] <- "<NA>"
  names(Fscore) <- nms

  # Create a data frame with all results
  res <- data.frame(
    Fscore = Fscore, Recall = Recall, Precision = Precision,
    Specificity = Specificity, NPV = NPV, FPR = FPR, FNR = FNR, FDR = FDR,
    FOR = FOR, LRPT = LRPT, LRNT = LRNT, LRPS = LRPS, LRNS = LRNS,
    BalAcc = BalAcc, MCC = MCC, Chisq = Chisq, Bray = Bray, Auto = TP_FP,
    Manu = TP_FN, A_M = Auto_Manu, TP = TP, FP = FP, FN = FN, TN = TN)

  lev <- rownames(object)
  lev[is.na(lev)] <- "<NA>"
  rownames(res) <- lev

  # Sort the table in function of one parameter... by default Fscore
  if (length(sort.by) && sort.by[1] != FALSE) {
    if (sort.by[1] %in% names(res)) {
      ord <- order(res[, sort.by], decreasing = decreasing)
      res <- res[ord, ]
      lev <- lev[ord]
    } else {
      warning("wrong sort.by: ignored and no sort performed")
    }
  }

  # What type of results should we return?
  if (length(type) && type[1] != "all") {
    okType <- type[type %in% names(res)]
    if (!length(okType))
      stop("Wrong type specified")
    if (length(okType) < length(type))
      warning("one or more wrong types are ignored")
    res <- res[, okType]
    # If the data are reduced to a numeric vector, reinject names
    # and return only this vector
    if (is.numeric(res)) {
      res <- as.numeric(res)
      names(res) <- lev
      attr(res, "stat.type") <- okType
      return(res)
    }
  }

  attr(res, "stats") <- attr(object, "stats")
  attr(res, "stats.weighted") <-
    c(error = Error, Fmicro = Fmicro, Fmacro = Fmacro)

  class(res) <- c("summary.confusion", class(res))
  res
}

#' @rdname confusion
#' @export
#' @method print summary.confusion
print.summary.confusion <- function(x, ...) {
  # General stats on the confusion matrix
  Stats <- attr(x, "stats")
  Error <- round(Stats["error"] * 100, 1)
  cat(Stats["total"], " items classified with ", Stats["truepos"],
    " true positives (error = ", Error, "%)\n",
    sep = "")
  cat("\nGlobal statistics on reweighted data:\n")
  Stats2 <- attr(x, "stats.weighted")
  cat("Error rate: ", round(Stats2["error"] * 100, digits = 1),
    "%, F(micro-average): ", round(Stats2["Fmicro"], digits = 3),
    ", F(macro-average): ", round(Stats2["Fmacro"], digits = 3), "\n\n",
    sep = "")
  X <- x
  class(X) <- class(X)[-1]
  print(X)

  invisible(x)
}
