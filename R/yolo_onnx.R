#' Create an ONNX-backed YOLO detector (Ultralytics export)
#'
#' @param onnx_file Path to an exported ONNX model (recommended: exported with nms=True).
#' @param class_labels Character vector of class labels (index -> label).
#' @param input_size Model input size (typically 640).
#' @param providers Execution providers in priority order, e.g. c("CUDAExecutionProvider","CPUExecutionProvider").
#' @export
ovml_yolo_onnx <- function(
        version = "11n-coco-nms",
        weights_file = "auto",
        providers = c("CPUExecutionProvider")
) {
    reg <- .ovml_onnx_registry()
    if (!version %in% reg$version) {
        stop(
            "Unknown ONNX version: ", version,
            "\nAvailable: ", paste(reg$version, collapse = ", ")
        )
    }
    row <- reg[match(version, reg$version), ]
    
    # Resolve weights path
    if (identical(weights_file, "auto")) {
        # Follow the same pattern you already use: cache dir + download + hash verify
        cache_dir <- ovml_cache_dir()              # already exists in ovml
        dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
        #dest <- file.path(cache_dir, paste0("ovml_", version, ".onnx"))
        
        # This should match how ovml currently downloads weights.
        # If you already have ovml_download_if(url, dest, sha1=...), reuse it.
        weights_file <- ovml_download_if(
            url = row$weights_url,
            #dest = dest,
            expected_sha1 = row$sha1
        )
    } else {
        if (!file.exists(weights_file)) stop("weights_file not found: ", weights_file)
    }
    
    # Resolve labels
    class_labels <- ovml_class_labels(row$labels)  # if you have this helper; else implement it
    providers <- as.list(providers)   # IMPORTANT
    
    structure(
        list(
            version = version,
            onnx_file = normalizePath(weights_file, winslash = "/", mustWork = TRUE),
            class_labels = class_labels,
            input_size = as.integer(row$input_size),
            providers = providers,
            nms_in_model = isTRUE(row$nms_in_model)
        ),
        class = "ovml_yolo_onnx"
    )
}


#' List available ONNX model versions
#' @export
ovml_onnx_versions <- function() {
    reg <- .ovml_onnx_registry()
    reg[, c("version", "labels", "input_size", "nms_in_model")]
}

#' Install Python dependencies for ONNX inference
#' @export
ovml_install_onnxruntime <- function(gpu = FALSE) {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop("Package 'reticulate' is required. Install it first.")
    }
    pkg <- if (isTRUE(gpu)) "onnxruntime-gpu" else "onnxruntime"
    reticulate::py_install(pkg, pip = TRUE)
    invisible(TRUE)
}


#' @export
ovml_yolo_detect.ovml_yolo_onnx <- function(
        net,
        image_file,
        conf = 0.25,
        classes = NULL,
        ...
) {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop("reticulate is required for ONNX inference.")
    }
    ort <- reticulate::import("onnxruntime", delay_load = TRUE, convert = FALSE)
    np <- reticulate::import("numpy", delay_load = TRUE, convert = FALSE)
    
    # --- load image (match ovml style: magick used already in ovml)
    if (!requireNamespace("magick", quietly = TRUE)) {
        stop("magick is required for image loading.")
    }
    img <- magick::image_read(image_file)
    
    # Get original dimensions
    info <- magick::image_info(img)
    orig_w <- info$width
    orig_h <- info$height
    
    # --- letterbox resize to net$input_size (keep aspect)
    # This mirrors what YOLO expects. Implementation below is minimal but works.
    target <- net$input_size
    scale <- min(target / orig_w, target / orig_h)
    new_w <- as.integer(round(orig_w * scale))
    new_h <- as.integer(round(orig_h * scale))
    
    img_rs <- magick::image_resize(img, paste0(new_w, "x", new_h))
    
    # Create padded canvas
    pad_x <- target - new_w
    pad_y <- target - new_h
    pad_left <- floor(pad_x / 2)
    pad_top  <- floor(pad_y / 2)
    
    canvas <- magick::image_blank(width = target, height = target, color = "black")
    img_pad <- magick::image_composite(
        canvas,
        img_rs,
        operator = "over",
        offset = paste0("+", pad_left, "+", pad_top)
    )
    
    # NOTE: image_extent pads at bottom/right by default depending on gravity.
    # For precise symmetric padding, you can composite onto a blank canvas; this MVP is OK.
    # If you want exact parity with Ultralytics letterbox, we’ll refine in the next step.
    
    # --- convert to numeric array (HWC), normalize 0..1, then NCHW
    arr <- magick::image_data(img_pad, channels = "rgb")
    # magick returns raw [0..255] with dims [channels, width, height]
    # Convert to float32 NCHW [1,3,H,W]
    arr_num <- array(as.integer(arr), dim = dim(arr))
    
    # Convert [C, W, H] -> [C, H, W]
    chw <- aperm(arr_num, c(1, 3, 2))
    
    # Add batch dim -> [1, C, H, W]
    x <- array(chw, dim = c(1L, dim(chw)))
    
    #x[ , c(3,2,1), , ] <- x[ , c(1,2,3), , ]   # (if x is [1,3,H,W])
    
    # --- create session
    providers <- reticulate::r_to_py(as.list(net$providers))
    sess <- ort$InferenceSession(net$onnx_file, providers = providers)
    inputs <- reticulate::py_to_r(sess$get_inputs())
    input_name <- inputs[[1]]$name   # python list index (0-based)
    #input_name <- reticulate::py_to_r(input0$name)
    
    
    x_np <- np$asarray(reticulate::r_to_py(x), dtype = np$float32)
    x_np <- np$ascontiguousarray(x_np)
    
    # --- run inference
    dtype <- reticulate::py_to_r(x_np$dtype$name)
    if (!identical(dtype, "float32")) stop("ONNX input dtype is ", dtype, " (expected float32)")
    
    out <- reticulate::py_to_r(sess$run(NULL, setNames(list(x_np), input_name)))
    y <- out[[1]]

    browser()
    
    d <- dim(y)
    if (length(d) == 3L) {
        y <- y[1,,]  # batch dimension
    }
    
    # y columns: x1,y1,x2,y2,score,class
    if (ncol(y) < 6L) stop("Unexpected ONNX output shape; expected last dim >= 6.")
    
    det <- data.frame(
        xmin = y[,1],
        ymin = y[,2],
        xmax = y[,3],
        ymax = y[,4],
        conf = y[,5],
        class_id = as.integer(y[,6]) + 1L,  # many exports are 0-indexed; adjust if needed
        stringsAsFactors = FALSE
    )
    
    # filter confidence + classes
    det <- det[is.finite(det$conf) & det$conf >= conf, , drop = FALSE]
    if (!is.null(classes)) {
        # classes can be numeric IDs (1-based) or labels
        if (is.character(classes)) {
            keep <- net$class_labels[det$class_id] %in% classes
        } else {
            keep <- det$class_id %in% as.integer(classes)
        }
        det <- det[keep, , drop = FALSE]
    }
    
    # map labels
    det$class <- net$class_labels[pmax(pmin(det$class_id, length(net$class_labels)), 1L)]
    
    # --- rescale boxes back to original image coordinates
    # reverse letterbox scaling
    # NOTE: because padding handling above is approximate, this rescale is the “MVP”.
    # In the next PR we’ll make letterbox+unletterbox exact.
    det$xmin <- (det$xmin - pad_left) / scale
    det$xmax <- (det$xmax - pad_left) / scale
    det$ymin <- (det$ymin - pad_top) / scale
    det$ymax <- (det$ymax - pad_top) / scale
    
    # clip to image bounds
    det$xmin <- pmax(0, pmin(det$xmin, orig_w))
    det$xmax <- pmax(0, pmin(det$xmax, orig_w))
    det$ymin <- pmax(0, pmin(det$ymin, orig_h))
    det$ymax <- pmax(0, pmin(det$ymax, orig_h))
    
    # --- adapt to ovml’s existing output conventions
    # If ovml currently uses columns like image_number/box_number/etc.,
    # add them here to match exactly (easy once we inspect current output).
    det$image_number <- 1L
    det$box_number <- seq_len(nrow(det))
    
    det <- det[is.finite(det$conf) & det$conf > 0, , drop = FALSE]
    det <- det[det$xmax > det$xmin & det$ymax > det$ymin, , drop = FALSE]
    
    det[, c("image_number","box_number","class","conf","xmin","ymin","xmax","ymax","class_id")]
}


#' Inspect ONNX model input/output names and shapes
#' @export
ovml_onnx_inspect <- function(onnx_file) {
    ort <- reticulate::import("onnxruntime", delay_load = TRUE)
    sess <- ort$InferenceSession(onnx_file, providers = c("CPUExecutionProvider"))
    ins <- sess$get_inputs()
    outs <- sess$get_outputs()
    list(
        inputs = lapply(ins, function(i) list(name=i$name, shape=i$shape, type=i$type)),
        outputs = lapply(outs, function(o) list(name=o$name, shape=o$shape, type=o$type))
    )
}

.ovml_onnx_registry <- function() {
    data.frame(
        version = c("11n-coco-nms", "11n-mvb"),
        weights_url = c(
            "https://github.com/openvolley/ovml/releases/download/v0.1.0/yolo11n-coco-nms.onnx",
            "https://github.com/openvolley/ovml/releases/download/models/yolo11n-mvb-nms.onnx"
        ),
        sha1 = c(
            "29037c19f597a152e3911e267ae6e994a623b770",
            "PUT_SHA256_HERE"
        ),
        input_size = c(640L, 640L),
        labels = c("coco", "mvb"),
        nms_in_model = c(TRUE, TRUE),
        stringsAsFactors = FALSE
    )
}
