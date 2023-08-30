#' Get or set priors on a confusion matrix
#'
#' @description
#' Most metrics in supervised classifications are sensitive to the relative
#' proportion of the items in the different classes. When a confusion matrix is
#' calculated on a test set, it uses the proportions observed on that test set.
#' If they are representative of the proportions in the population, metrics are
#' not biased. When it is not the case, priors of a **confusion** object can be
#' adjusted to better reflect proportions that are supposed to be observed in
#' the different classes in order to get more accurate metrics.
#'
#' @param object a **confusion** object (or another class if a method is
#'   implemented)
#' @param ... further arguments passed to methods
#' @param value a (named) vector of positive numbers of zeros of
#'   the same length as the number of classes in the **confusion** object. It
#'   can also be a single >= 0 number and in this case, equal probabilities are
#'   applied to all the classes (use 1 for relative frequencies and 100 for
#'   relative frequencies in percent). If the value has zero length or is
#'   `NULL`, original prior probabilities (from the test set) are used. If the
#'   vector is named, names must correspond to existing class names in the
#'   **confusion** object.
#'
#' @return  [prior()] returns the current class frequencies associated with
#'   the first classification tabulated in the **confusion** object, i.e., for
#'   rows in the confusion matrix.
#' @export
#' @seealso [confusion()]
#' @examples
#' data("Glass", package = "mlbench")
#' # Use a little bit more informative labels for Type
#' Glass$Type <- as.factor(paste("Glass", Glass$Type))
#' # Use learning vector quantization to classify the glass types
#' # (using default parameters)
#' summary(glass_lvq <- ml_lvq(Type ~ ., data = Glass))
#'
#' # Calculate cross-validated confusion matrix
#' (glass_conf <- confusion(cvpredict(glass_lvq), Glass$Type))
#'
#' # When the probabilities in each class do not match the proportions in the
#' # training set, all these calculations are useless. Having an idea of
#' # the real proportions (so-called, priors), one should first reweight the
#' # confusion matrix before calculating statistics, for instance:
#' prior1 <- c(10, 10, 10, 100, 100, 100) # Glass types 1-3 are rare
#' prior(glass_conf) <- prior1
#' glass_conf
#' summary(glass_conf, type = c("Fscore", "Recall", "Precision"))
#'
#' # This is very different than if glass types 1-3 are abundants!
#' prior2 <- c(100, 100, 100, 10, 10, 10) # Glass types 1-3 are abundants
#' prior(glass_conf) <- prior2
#' glass_conf
#' summary(glass_conf, type = c("Fscore", "Recall", "Precision"))
#'
#' # Weight can also be used to construct a matrix of relative frequencies
#' # In this case, all rows sum to one
#' prior(glass_conf) <- 1
#' print(glass_conf, digits = 2)
#' # However, it is easier to work with relative frequencies in percent
#' # and one gets a more compact presentation
#' prior(glass_conf) <- 100
#' glass_conf
#'
#' # To reset row class frequencies to original propotions, just assign NULL
#' prior(glass_conf) <- NULL
#' glass_conf
#' prior(glass_conf)
prior <- function(object, ...)
  UseMethod("prior")

#' @rdname prior
#' @export
#' @method prior confusion
prior.confusion <- function(object, ...)
  attr(object, "prior")

#' @rdname prior
#' @export
`prior<-` <- function(object, ..., value)
  UseMethod("prior<-")

#' @rdname prior
#' @export
#' @method prior<- confusion
`prior<-.confusion` <- function(object, ..., value) {
  rsums <- rowSums(object)
  if (!length(value)) {# value is NULL or of zero length
    # Reset prior to original frequencies
    value <- attr(object, "row.freqs")
    res <- round(object / rsums * value)

  } else if (is.numeric(value)) {# value is numeric

    if (length(value) == 1) { # value is a single number
      if (is.na(value) || !is.finite(value) || value <= 0)
        stop("value must be a finite positive number")
      res <- object / rsums * as.numeric(value)

    } else {# value is a vector of numerics
      # It must be either of the same length as nrow(object) or of
      # levels(objects)
      l <- length(value)
      n <- names(value)
      l2 <- levels(object)

      if (l == nrow(object)) {
        # If the vector is named, check names and possibly reorder it
        if (length(n))
          if (all(n %in% rownames(object))) {
            value <- value[rownames(object)]
          } else {
            stop("Names of the values do not match levels in the confusion matrix")
          }

      } else if (l == length(l2)) {
        # Assume names as levels(object), if they are not provides
        if (!length(n)) names(value) <- n <- l2

        # If the vector is named, check names match levels
        if (length(n))
          if (all(n %in% l2)) {
            # Extract levels used in the confusion matrix
            value <- value[rownames(object)]
          } else {
            stop("Names of the values do not match levels in the confusion matrix")
          }

      } else {
        stop("length of 'value' do not match the number of levels in the confusion matrix")
      }

      res <- object / rsums * as.numeric(value)
    }

  } else {
    stop("value must be a numeric vector, a single number or NULL")
  }

  attr(res, "prior") <- value
  # Take care to rows with no items! => put back zeros!
  res[rsums == 0] <- 0
  res
}
