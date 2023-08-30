#' Plot a confusion matrix
#'
#' @description
#' Several graphical representations of **confusion** objects are possible: an
#' image of the matrix with colored squares, a barplot comparing recall and
#' precision, a stars plot also comparing two metrics, possibly also comparing
#' two different classifiers of the same dataset, or a dendrogram grouping the
#' classes relative to the errors observed in the confusion matrix (classes
#' with more errors are pooled together more rapidly).
#'
#' @param x a **confusion** object
#' @param y `NULL` (not used), or a second **confusion** object when two
#'   different classifications are compared in the plot (`"stars"` type).
#' @param type the kind of plot to produce (`"image"`, the default, or
#'   `"barplot"`, `"stars"`, `"dendrogram"`).
#' @param stat1 the first metric to plot for the `"stars"` type (Recall by
#'    default).
#' @param stat2 the second metric to plot for the `"stars"` type (Precision by
#' default).
#' @param names names of the two classifiers to compare
#' @param ... further arguments passed to the function. It can be all arguments
#'   or the corresponding plot.
#' @param labels labels to use for the two classifications. By default, they are
#'   the same as `vars`, or the one in the confusion matrix.
#' @param sort are rows and columns of the confusion matrix sorted so that
#'   classes with larger confusion are closer together? Sorting is done
#'   using a hierarchical clustering with [hclust()]. The clustering method
#'   is `"ward.D2"` by default, but see the [hclust()] help for other options).
#'   If `FALSE` or `NULL`, no sorting is done.
#' @param numbers are actual numbers indicated in the confusion matrix image?
#' @param digits the number of digits after the decimal point to print in the
#' confusion matrix. The default or zero leads to most compact presentation
#' and is suitable for frequencies, but not for relative frequencies.
#' @param mar graph margins.
#' @param cex text magnification factor.
#' @param cex.axis  idem for axes. If `NULL`, the axis is not drawn.
#' @param cex.legend idem for legend text. If `NULL`, no legend is added.
#' @param asp graph aspect ratio. There is little reasons to change the default
#' value of 1.
#' @param col color(s) to use for the plot.
#' @param colfun a function that calculates a series of colors, like e.g.,
#'   [cm.colors()] that accepts one argument being the number of colors
#'   to be generated.
#' @param ncols the number of colors to generate. It should preferably be
#'  2 * number of levels + 1, where levels is the number of frequencies you
#'  want to evidence in the plot. Default to 41.
#' @param col0 should null values be colored or not (no, by default)?
#' @param grid.col color to use for grid lines, or `NULL` for not drawing grid
#'   lines.
#' @param main main title of the plot.
#' @param min.width minimum bar width required to add numbers.
#'
#' @return Data calculate to create the plots are returned invisibly. These
#'   functions are mostly used for their side-effect of producing a plot.
#' @export
#'
#' @examples
#' data("Glass", package = "mlbench")
#' # Use a little bit more informative labels for Type
#' Glass$Type <- as.factor(paste("Glass", Glass$Type))
#'
#' # Use learning vector quantization to classify the glass types
#' # (using default parameters)
#' summary(glass_lvq <- ml_lvq(Type ~ ., data = Glass))
#'
#' # Calculate cross-validated confusion matrix and plot it in different ways
#' (glass_conf <- confusion(cvpredict(glass_lvq), Glass$Type))
#' # Raw confusion matrix: no sort and no margins
#' print(glass_conf, sums = FALSE, sort = FALSE)
#' # Plots
#' plot(glass_conf) # Image by default
#' plot(glass_conf, sort = FALSE) # No sorting
#' plot(glass_conf, type = "barplot")
#' plot(glass_conf, type = "stars")
#' plot(glass_conf, type = "dendrogram")
#'
#' # Build another classifier and make a comparison
#' summary(glass_naive_bayes <- ml_naive_bayes(Type ~ ., data = Glass))
#' (glass_conf2 <- confusion(cvpredict(glass_naive_bayes), Glass$Type))
#'
#' # Comparison plot for two classifiers
#' plot(glass_conf, glass_conf2)
plot.confusion <- function(x, y = NULL,
  type = c("image", "barplot", "stars", "dendrogram"), stat1 = "Recall",
  stat2 = "Precision", names, ...) {
  if (is.null(y)) {
    type <- match.arg(type)[1]
  } else{
    type <- "stars"
  }
  if (missing(names))
    names <- c(substitute(x), substitute(y))
  res <- switch(type,
    image = confusion_image(x, y, ...),
    barplot = confusion_barplot(x, y, ...),
    stars = confusion_stars(x, y, stat1 = stat1, stat2 = stat2, names, ...),
    dendrogram = confusion_dendrogram(x, y, ...),
    stop("'type' must be 'image', 'barplot', 'stars' or 'dendrogram'"))
  invisible(res)
}

#' @rdname plot.confusion
#' @export
confusion_image <- function(x, y = NULL, labels = names(dimnames(x)),
  sort = "ward.D2", numbers = TRUE, digits = 0, mar = c(3.1, 10.1, 3.1, 3.1),
  cex = 1, asp = 1, colfun, ncols = 41, col0 = FALSE, grid.col = "gray", ...) {
  if (!inherits(x, "confusion"))
    stop("'x' must be a 'confusion' object")

  if (!is.null(y))
    stop("cannot use a second classifier 'y' for this plot")

  # Default labels in case none provided
  if (is.null(labels)) labels <- c("Actual", "Predicted")

  # Default color function
  # (greens for correct values, reds for errors, white for zero)
  if (missing(colfun)) {
    colfun <- function(n, alpha = 1, s = 0.9, v = 0.9) {
      if ((n <- as.integer(n[1L])) <= 0)
        return(character(0L))
      # Initial (red) and final (green) colors with white in between
      cols <- c(hsv(h = 0, s = s, v = v, alpha = alpha),   # Red
        hsv(h = 0, s = 0, v = v, alpha = alpha),   # White
        hsv(h = 2/6, s = s, v = v, alpha = alpha)) # Green
      # Use a color ramp from red to white to green
      colorRampPalette(cols)(n)
    }
  }

  n <- ncol(x)

  # Do we sort items?
  if (length(sort) && any(!is.na(sort)) && any(sort != FALSE) &&
      any(sort != "")) {
    # Grouping of items
    confuSim <- x + t(x)
    confuSim <- 1 - (confuSim / sum(confuSim) * 2)
    confuDist <- structure(confuSim[lower.tri(confuSim)], Size = n,
      Diag = FALSE, Upper = FALSE, method = "confusion", call = "",
      class = "dist")
    order <- hclust(confuDist, method = sort)$order
    x <- x[order, order]
  }

  # Recode row and column names for more compact display
  colnames(x) <- names2 <- formatC(1:n, digits = 1, flag = "0")
  rownames(x) <- names1 <- paste(rownames(x), names2)

  # Transform for better colorization
  # (use a transfo to get 0, 1, 2, 3, 4, 7, 10, 15, 25+)
  confuCol <- x
  confuCol <- log(confuCol + .5) * 2.33
  confuCol[confuCol < 0] <- if (isTRUE(as.logical(col0))) 0 else NA
  confuCol[confuCol > 10] <- 10

  # Negative values (in green) on the diagonal (correct IDs)
  diag(confuCol) <- -diag(confuCol)

  # Make an image of this matrix
  opar <- par(no.readonly = TRUE)
  on.exit(par(opar))
  par(mar = mar, cex = cex)
  image(1:n, 1:n, -t(confuCol[nrow(confuCol):1, ]), zlim = c(-10, 10),
    asp = asp, bty = "n", col = colfun(ncols), xaxt = "n", yaxt = "n",
    xlab = "", ylab = "", ...)

  # Indicate the actual numbers
  if (isTRUE(as.logical(numbers))) {
    confuTxt <- as.character(round(x[n:1, ], digits = digits))
    confuTxt[confuTxt == "0"] <- ""
    text(rep(1:n, each = n), 1:n, labels = confuTxt)
  }

  # Add the grid
  if (length(grid.col)) {
    abline(h = 0:n + 0.5, col = grid.col)
    abline(v = 0:n + 0.5, col = grid.col)
  }

  # Add the axis labels
  axis(1, 1:n, labels = names2, tick =  FALSE, padj = 0)
  axis(2, 1:n, labels = names1[n:1], tick =  FALSE, las = 1, hadj = 1)
  axis(3, 1:n, labels = names2, tick =  FALSE)
  axis(4, 1:n, labels = names2[n:1], tick =  FALSE, las = 1, hadj = 0)

  # Add labels at top-left
  if (length(labels)) {
    if (length(labels) != 2)
      stop("You must provide two labels")
    mar[2] <- 1.1
    par(mar = mar, new = TRUE)
    plot(0, 0, type = "n", xaxt = "n", yaxt = "n", bty = "n")
    mtext(paste(labels, collapse = " // "), adj = 0, line = 1, cex = cex)
  }

  # Return the confusion matrix, as displayed, in text format
  invisible(x)
}

#' @rdname plot.confusion
#' @export
confusionImage <- confusion_image

# Confusion barplot with recall and precision in green bars
# TODO: various bar rescaling possibilities!!!
#' @rdname plot.confusion
#' @export
confusion_barplot <- function(x, y = NULL,
  col = c("PeachPuff2", "green3", "lemonChiffon2"), mar = c(1.1, 8.1, 4.1, 2.1),
  cex = 1, cex.axis = cex, cex.legend = cex,
  main = "F-score (precision versus recall)", numbers = TRUE, min.width = 17,
  ...) {
  if (!inherits(x, "confusion"))
    stop("'x' must be a 'confusion' object")

  if (!is.null(y))
    stop("cannot use a second classifier 'y' for this plot")

  # F-score is 2 * recall * precision / (recall + precision), ... but also
  # F-score = TP / (TP + FP/2 + FN/2). We represent this in a barplot
  TP <- tp <- diag(x)
  FP <- fp <- colSums(x) - tp
  FN <- fn <- rowSums(x) - tp
  # In case we have missing data...
  fn[is.na(tp)] <- 50
  fp[is.na(tp)] <- 50
  tp[is.na(tp)] <- 0

  # We scale these values, so that the sum fp/2 + tp + fn/2 makes 100
  scale <- fp/2 + tp + fn/2
  res <- matrix(c(fp/2 / scale * 100, tp / scale * 100, fn/2 / scale * 100),
    ncol = 3)
  colnames(res) <- c("FPcontrib", "Fscore", "FNcontrib") # In %
  Labels <- names(attr(x, "col.freqs"))

  # The graph is ordered in decreasing F-score values
  pos <- order(res[, 2], decreasing = TRUE)
  res <- res[pos, ]
  FN <- FN[pos]
  FP <- FP[pos]
  TP <- TP[pos]
  Labels <- Labels[pos]
  l <- length(FN)

  # Plot the graph
  omar <- par("mar")
  on.exit(par(omar))
  par(mar = mar)
  # The barplot
  barplot(t(res), horiz = TRUE, col = col, xaxt = "n", las = 1, space = 0,
    main = main, ...)
  # The line that shows where symmetry is
  lines(c(50, 50), c(0, l), lwd = 1)

  # Do we add figures into the plot?
  if (isTRUE(as.logical(numbers))) {
    # F-score is written in the middle of the central bar
    xpos <- res[, 1] + res[, 2] / 2
    text(xpos, 1:l - 0.5, paste("(", round(res[, 2]), "%)", sep = ""),
      adj = c(0.5, 0.5), cex = cex)

    # Add the number of FP and FN to the left and right, respectively
    text(rep(1, l), 1:l - 0.5, round(FP), adj = c(0, 0.5), cex = cex)
    text(rep(99, l), 1:l - 0.5, round(FN), adj = c(1, 0.5), cex = cex)
  }

  # Add a legend (if cex.legend is not NULL)
  if (length(cex.legend)) {
    legend(50, l * 1.05, legend = c("False Positives",
      "2*TP (F-score %)", "False Negatives"), cex = cex.legend, xjust = 0.5, yjust = 1,
      fill = col, bty = "n", horiz = TRUE)
  }

  # Add axes if cex.axis is not NULL
  if (length(cex.axis))
    axis(2, 1:l - 0.5, tick = FALSE, las = 1, cex.axis = cex.axis,
      labels = Labels)

  invisible(res)
}

#' @rdname plot.confusion
#' @export
confusionBarplot <- confusion_barplot

# TODO: check the box around the legend
#' @rdname plot.confusion
#' @export
confusion_stars <- function(x, y = NULL, stat1 = "Recall", stat2 = "Precision",
  names, main, col = c("green2", "blue2", "green4", "blue4"), ...) {
  # Check objects
  if (!inherits(x, "confusion"))
    stop("'x' must be a 'confusion' object")
  if (!is.null(y) && !inherits(x, "confusion"))
    stop("'y' must be NULL or a 'confusion' object")

  # Check stats
  SupportedStats <- c("Recall", "Precision", "Specificity",
    "NPV", "FPR", "FNR", "FDR", "FOR")
  stat1 <- stat1[1]
  if (!stat1 %in% SupportedStats)
    stop("stats1 must be one of Recall, Precision, Specificity, NPV, FPR, FNR, FDR, FOR")
  stat2 <- stat2[1]
  if (!stat2 %in% SupportedStats)
    stop("stats2 must be one of Recall, Precision, Specificity, NPV, FPR, FNR, FDR, FOR")

  # Choose colors TODO: add a colors argument!
  Blue <- topo.colors(16)
  Green <- terrain.colors(16)
  Stat <- summary(x)
  if (!is.null(y)) { # Comparison of two confusion matrices
    Stat2 <- summary(y)
    Data <- data.frame(Stat2[, stat1], Stat[, stat1], Stat[, stat2],
      Stat2[, stat2])
    Data <- rbind(Data, rep(0, 4))
    colnames(Data) <- paste(rep(c(stat1, stat2), each = 2), c(2, 1, 1, 2))
    if (missing(main)) {# Calculate a suitable title
      if (missing(names)) {
        names <- c(substitute(x), substitute(y))
      } else if (length(names) != 2) {
        stop("you must provide two nmaes for the two compared classifiers")
      }
      names <- as.character(names)
      main <- paste("Groups comparison (1 =", names[1], ", 2 =", names[2], ")")
    }
    if (length(col) >= 4) {
      col <- col[c(3, 1, 2, 4)]
    } else {
      stop("you must provide four colors for the two statistics and the two classifiers")
    }
  } else {# Single confusion matrix
    Data <- data.frame(Stat[, stat1], Stat[, stat2])
    Data <- rbind(Data, rep(0, 2))
    colnames(Data) <- c(stat1, stat2)
    if (missing(main))
      main <- paste("Groups comparison")
    if (length(col) >= 2) {
      col <- col[1:2]
    } else {
      stop("you must provide two colors for the two statistics")
    }
  }
  rownames(Data) <- c(rownames(Stat), " ")
  # Note: last one is empty box for legend

  # Save graph parameters and restore on exit
  opar <- par(no.readonly = TRUE)
  on.exit(par(opar))

  # Calculate key location
  kl <- stars(Data, draw.segments = TRUE, scale = FALSE,
    len = 0.8, main = main, col.segments = col, plot = FALSE, ...)
  kcoords <- c(max(kl[, 1]), min(kl[, 2]))
  kspan <- apply(kl, 2, min) / 1.95

  # Draw the plot
  res <- stars(Data, draw.segments = TRUE, scale = FALSE, key.loc = kcoords,
    len = 0.8, main = main, col.segments = col, ...)

  # Draw a rectangle around key to differentiate it from the rest
  rect(kcoords[1] - kspan[1], kcoords[2] - kspan[2], kcoords[1] + kspan[1],
    kcoords[2] + kspan[2])

  res
}

#' @rdname plot.confusion
#' @export
confusionStars <- confusion_stars

#' @rdname plot.confusion
#' @export
confusion_dendrogram <- function(x, y = NULL, labels = rownames(x),
  sort = "ward.D2", main = "Groups clustering", ...) {
  # Check objects
  if (!inherits(x, "confusion"))
    stop("'x' must be a 'confusion' object")
  if (!is.null(y))
    stop("cannot use a second classifier 'y' for this plot")

  # Transform the confusion matrix into a symmetric matrix
  ConfuSim <- x + t(x)
  ConfuSim <- 1 - (ConfuSim / sum(ConfuSim) * 2)


  # Create the structure of a "dist" object
  ConfuDist <- structure(ConfuSim[lower.tri(ConfuSim)], Size = nrow(x),
    Diag = FALSE, Upper = FALSE, method = "confusion", call = "",
    class = "dist")

  # method :"ward.D2", "single", "complete", "average", "mcquitty",
  # "median" or "centroid"
  HC <- hclust(ConfuDist, method = as.character(sort)[1])
  plot(HC, labels = labels, main = main, ...)

  invisible(HC)
}

#' @rdname plot.confusion
#' @export
confusionDendrogram <- confusion_dendrogram
