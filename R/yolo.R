#' Construct YOLO network
#'
#' @references https://github.com/pjreddie/darknet, https://github.com/WongKinYiu/yolov7
#' @param version integer or string: one of
#' - 3 : YOLO v3
#' - 4 : YOLO v4
#' - "4-tiny" : YOLO v4-tiny
#' - "4-mvb" : an experimental network trained specifically to detect (only) volleyballs
#' - "4-tiny-mvb" : the v4-tiny version of the same
#' - 7 or "7-tiny" : YOLO v7 or v7-tiny
#'
#' @param device string: "cpu" or "cuda"
#' @param weights_file string: either the path to the weights file that already exists on your system or "auto". If "auto", the weights file will be downloaded if necessary and stored in the directory given by [ovml_cache_dir()]
#' @param class_labels character: the class labels used for network training. If missing or NULL, these default to `ovml_class_labels("coco")` for all models except "mvb" models, which use `ovml_class_labels("mvb")`
#'
#' @return A YOLO network object
#'
#' @examples
#' \dontrun{
#'   dn <- ovml_yolo()
#'   img <- ovml_example_image()
#'   res <- ovml_yolo_detect(dn, img)
#'   ovml_ggplot(img, res)
#' }
#'
#' @export
ovml_yolo <- function(version = 4, device = "cuda", weights_file = "auto", class_labels) {
    if (is.numeric(version)) version <- as.character(version)
    assert_that(version %in% c("3", "4", "4-tiny", "4-mvb", "4-tiny-mvb", "7", "7-tiny", "7-mvb", "7-tiny-mvb"))
    assert_that(is.string(device))
    device_was_specified <- !missing(device)
    device <- tolower(device)
    device <- match.arg(device, c("cpu", "cuda"))
    if (device == "cuda" && !cuda_is_available()) {
        if (device_was_specified) warning("'cuda' device not available, using 'cpu'")
        device <- "cpu"
    }
    to_cuda <- device == "cuda"
    device <- torch_device(device)
    expected_sha1 <- NULL
    from_jit <- FALSE
    if (version == "3") {
        if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("coco")
        dn <- yolo3_darknet(system.file(paste0("extdata/yolo/yolov", version, ".cfg"), package = "ovml"), device = device)
        w_url <- "https://pjreddie.com/media/files/yolov3.weights"
        expected_sha1 <- "520878f12e97cf820529daea502acca380f1cb8e"
    } else if (version %in% c("7", "7-mvb", "7-tiny", "7-tiny-mvb")) {
        if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) {
            class_labels <- ovml_class_labels(if (grepl("mvb", version)) "mvb" else "coco")
        }
        if (version == "7") {
            w_url <- "https://github.com/openvolley/ovml/releases/download/v0.1.0/yolov7.torchscript.pt"
            expected_sha1 <- "d8da940cd8175c2c670ad5ac86f5547b6f80c095"
        } else if (version == "7-tiny") {
            w_url <- "https://github.com/openvolley/ovml/releases/download/v0.1.0/yolov7-tiny.torchscript.pt"
            expected_sha1 <- "464a6f80b42b9800ff14d8693a218b4d25f36d31"
        } else if (version == "7-tiny-mvb") {
            w_url <- "https://github.com/openvolley/ovml/releases/download/v0.1.0/yolov7-tiny-mvb.torchscript.pt"
            expected_sha1 <- "5a398436b5929e6ee61cffc3dc0a9dc676ccd578"
        } else {
            ## "7-mvb"
            w_url <- "https://github.com/openvolley/ovml/releases/download/v0.1.0/yolov7-mvb.torchscript.pt"
            expected_sha1 <- "84b351f046bf6de1ea0eae74789b526d882c40f5"
        }
        from_jit <- TRUE
        dn <- NULL
    } else {
        dn <- yolo4_darknet(system.file(paste0("extdata/yolo/yolov", version, ".cfg"), package = "ovml"), device = device)
        if (version == "4") {
            if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("coco")
            w_url <- "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
            expected_sha1 <- "0143deb6c46fcc7f74dd35bf3c14edc3784e99ee"
        } else if (version == "4-tiny") {
            if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("coco")
            w_url <- "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
            expected_sha1 <- "451caaab22fb9831aa1a5ee9b5ba74a35ffa5dcb"
        } else if (version == "4-tiny-mvb") {
            if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("mvb")
            w_url <- "https://github.com/openvolley/ovml/releases/download/v0.0.7/yolov4-tiny-mvb.weights"
            expected_sha1 <- "8ef17c371ba0ee0a84e351a40eef83c44e049831"
        } else {
            if (missing(class_labels) || length(class_labels) < 1 || is.na(class_labels)) class_labels <- ovml_class_labels("mvb")
            w_url <- "https://github.com/openvolley/ovml/releases/download/v0.0.7/yolov4-mvb.weights"
            expected_sha1 <- "7ed27e4a3efd327cc04c596af784d682975d5a3e"
        }
    }
    if (length(weights_file) && nzchar(weights_file) && !is.na(weights_file)) {
        if (identical(tolower(weights_file), "auto")) {
            weights_file <- ovml_download_if(w_url, expected_sha1 = expected_sha1)
        }
        if (file.exists(weights_file)) {
            if (!from_jit) {
                dn$load_weights(weights_file)
            } else {
                dn <- torch::jit_load(weights_file)
                ## some bits to make other code work as-is
                dn$num_classes <- length(class_labels)
                dn$blocks <- list(list(height = 640L))
                dn$from_jit <- TRUE
            }
        } else {
            warning("weights file does not exist")
        }
    }
    dn$class_labels <- class_labels
    if (to_cuda) dn$to(device = device)
    dn$eval() ## set to inference mode
    dn
}

#' Detect objects in image using a YOLO network

#' @export
ovml_yolo_detect.default <- function(net, ...) {
    stop(
        "Unsupported detector object of class: ",
        paste(class(net), collapse = "/"),
        "\nCreate one with ovml_yolo() (torch) or ovml_yolo_onnx() (onnx)."
    )
}

#' @export
ovml_yolo_detect <- function(net, ...) {
    UseMethod("ovml_yolo_detect")
}

#' Processing of a video file requires that `ffmpeg` be installed on your system. [ovideo::ov_install_ffmpeg()] can help with this on Windows and Linux.
#'
#' @param net yolo: as returned by [ovml_yolo()]
#' @param image_file character: path to one or more image files, or a single video file (mp4, m4v, or mov extension)
#' @param conf scalar: confidence level
#' @param nms_conf scalar: non-max suppression confidence level
#' @param classes character: vector of class names, only detections of these classes will be returned
#' @param batch_size integer: the number of images to process as a batch. Increasing `batch_size` will make processing of multiple images faster, but requires more memory
#' @param ... : currently ignored
#'
#' @return A data.frame with columns "image_number", "image_file", "class", "score", "xmin", "xmax", "ymin", "ymax"
#'
#' @seealso [ovml_yolo()]
#'
#' @examples
#' \dontrun{
#'   dn <- ovml_yolo()
#'   img <- ovml_example_image()
#'   res <- ovml_yolo_detect(dn, img)
#'   ovml_ggplot(img, res)
#' }
#' @export
ovml_yolo_detect.ovml_yolo <- function(net, image_file, conf = 0.6, nms_conf = 0.4, classes, batch_size = 4, ...) {
    if (missing(classes)) classes <- NULL
    input_image_size <- as.integer(net$blocks[[1]]$height)
    if (length(input_image_size) < 1 || is.na(input_image_size) || input_image_size <= 0) stop("invalid input_image_size: ", input_image_size)
    if (length(net$num_classes) < 1 || is.na(net$num_classes)) stop("invalid number of classes")
    if (length(net$class_labels) != net$num_classes) stop("length of class_labels does not match the number of classes")
    if (any(grepl("\\.(mp4|m4v|mov)$", image_file, ignore.case = TRUE))) {
        if (length(image_file) == 1) {
            ## single video file, extract all frames
            image_file <- ovideo::ov_video_frames(image_file)
            ## could also use av::av_video_images ?
        } else {
            stop("only a single video file can be processed")
        }
    }
    starti <- seq(1, length(image_file), by = batch_size)
    endi <- pmin(starti + batch_size - 1L, length(image_file))
    do.call(rbind, lapply(seq_along(starti), function(i) {
        ##st <- system.time({
        this_image_files <- image_file[starti[i]:endi[i]]
        imgs <- lapply(this_image_files, function(im) {
            image <- image_read(im) ## h x w x rgb
            resized_image <- as.numeric(image_data(image_resz(image, input_image_size, preserve_aspect = YOLO_LETTERBOXING), "rgb"))
            list(tensor = torch_tensor(aperm(array(resized_image, dim = c(1, dim(resized_image))), c(1, 4, 2, 3)), device = net$device), original_wh = image_wh(image))
        })
        img_tensor <- torch_cat(lapply(imgs, function(z) z$tensor), dim = 1)
        ##}); cat("prep:\n"); print(st)
        ##if (net$device == "cuda") img_tensor <- img_tensor$to(device = torch_device("cuda"))
        ##st <- system.time({
            output <- net$forward(img_tensor)
        ##}); cat("inference:\n"); print(st)
        ##st <- system.time({
            if (isTRUE(net$from_jit)) output <- as.array(output[[1]]$to(device = torch_device("cpu"))) ## copy to cpu
        ##}); cat("data copy:\n"); print(st)
        owh <- do.call(rbind, lapply(imgs, function(z) z$original_wh))
        ##st <- system.time({
            res <- write_results(output, num_classes = net$num_classes, confidence = conf, nms_conf = nms_conf, original_wh = owh, input_image_size = input_image_size, class_labels = net$class_labels, classes = classes)
            res$image_file <- this_image_files[res$image_number]
            res$image_number <- as.integer(res$image_number + starti[i] - 1L)
        ##}); cat("results:\n"); print(st)
        res
    }))
}


#' @export
ovml_yolo_detect.nn_module <- function(net, image_file, conf = 0.6, nms_conf = 0.4,
                                       classes, batch_size = 4, ...) {
    # Guard: make sure this is an ovml YOLO-like net
    if (is.null(net$blocks) || is.null(net$num_classes) || is.null(net$class_labels)) {
        stop(
            "This nn_module does not look like an ovml YOLO detector.\n",
            "Expected fields: blocks, num_classes, class_labels."
        )
    }
    
    if (missing(classes)) classes <- NULL
    input_image_size <- as.integer(net$blocks[[1]]$height)
    if (length(input_image_size) < 1 || is.na(input_image_size) || input_image_size <= 0)
        stop("invalid input_image_size: ", input_image_size)
    if (length(net$num_classes) < 1 || is.na(net$num_classes)) stop("invalid number of classes")
    if (length(net$class_labels) != net$num_classes)
        stop("length of class_labels does not match the number of classes")
    
    if (any(grepl("\\.(mp4|m4v|mov)$", image_file, ignore.case = TRUE))) {
        if (length(image_file) == 1) {
            image_file <- ovideo::ov_video_frames(image_file)
        } else {
            stop("only a single video file can be processed")
        }
    }
    
    starti <- seq(1, length(image_file), by = batch_size)
    endi <- pmin(starti + batch_size - 1L, length(image_file))
    
    do.call(rbind, lapply(seq_along(starti), function(i) {
        this_image_files <- image_file[starti[i]:endi[i]]
        imgs <- lapply(this_image_files, function(im) {
            image <- magick::image_read(im)
            resized_image <- as.numeric(magick::image_data(
                image_resz(image, input_image_size, preserve_aspect = YOLO_LETTERBOXING),
                "rgb"
            ))
            list(
                tensor = torch::torch_tensor(
                    aperm(array(resized_image, dim = c(1, dim(resized_image))), c(1, 4, 2, 3)),
                    device = net$device
                ),
                original_wh = image_wh(image)
            )
        })
        
        img_tensor <- torch::torch_cat(lapply(imgs, `[[`, "tensor"), dim = 1)
        
        output <- net$forward(img_tensor)
        if (isTRUE(net$from_jit)) output <- as.array(output[[1]]$to(device = torch::torch_device("cpu")))
        
        owh <- do.call(rbind, lapply(imgs, `[[`, "original_wh"))
        
        res <- write_results(output, num_classes = net$num_classes, confidence = conf,
                             nms_conf = nms_conf, original_wh = owh, input_image_size = input_image_size,
                             class_labels = net$class_labels, classes = classes)
        
        res$image_file <- this_image_files[res$image_number]
        res$image_number <- as.integer(res$image_number + starti[i] - 1L)
        res
    }))
}


#' Run YOLO inference on a video (sampled frames + timestamps)
#'
#' This is a convenience wrapper around [ovideo::ov_video_frames()] and
#' [ovml_yolo_detect()], adding time-windowing, frame sampling and timestamps.
#'
#' @param net YOLO network as returned by [ovml_yolo()].
#' @param video_file Path to a video file.
#' @param start_time Numeric seconds from the start of the video (default 0).
#' @param end_time Numeric seconds from the start of the video (optional).
#' @param duration Numeric seconds (optional). If `end_time` is missing, `duration` is used.
#' @param fps_extract Numeric. Frames per second to extract (sampling rate). If `NULL`,
#'   all frames are extracted.
#' @param fps_video Numeric. Video framerate. If missing, inferred via `av::av_video_info()`.
#'   Used only to compute timestamps when `fps_extract` is `NULL`.
#' @param conf,nms_conf,classes,batch_size Passed to [ovml_yolo_detect()].
#' @param outdir Output directory for extracted frames. If missing, a temp dir is used.
#' @param keep_frames Logical; if `FALSE`, extracted frames are deleted on exit.
#' @param format "jpg" or "png" (passed to [ovideo::ov_video_frames()]).
#' @param jpg_quality Numeric 1-31 (passed to [ovideo::ov_video_frames()] if `format="jpg"`).
#' @param debug Logical; passed to [ovideo::ov_video_frames()].
#'
#' @return A list with elements: `detections`, `frames`, `meta`.
#' 
#' @seealso [ovml_yolo()]
#'
#' @examples
#' \dontrun{
#'   library(ovideo)
#'   v <- ov_example_video(1)
#'   dn <- ovml_yolo()
#'   out <- ovml_yolo_detect_video(dn, v, start_time = 0, duration = 3, fps_extract = 5)
#'   head(out$detections)
#' }
#' @export
ovml_yolo_detect_video <- function(
        net,
        video_file,
        start_time = 0,
        end_time,
        duration,
        fps_extract = 5,
        fps_video,
        conf = 0.6,
        nms_conf = 0.4,
        classes,
        batch_size = 4,
        outdir,
        keep_frames = FALSE,
        format = "jpg",
        jpg_quality = 1,
        debug = FALSE
) {
    stopifnot(length(video_file) == 1, is.character(video_file))
    if (!file.exists(video_file)) stop("video_file does not exist: ", video_file)
    
    # Derive end_time/duration logic
    if (missing(end_time) && missing(duration)) {
        # default: process to end of file? We can’t reliably know duration without probing,
        # so require one. You can change this to probe via av if you prefer.
        stop("Provide either end_time or duration.")
    }
    if (missing(end_time)) end_time <- start_time + duration
    if (!missing(duration)) end_time <- start_time + duration
    
    if (missing(outdir)) {
        outdir <- tempfile("ovml_frames_")
        dir.create(outdir, recursive = TRUE)
    } else {
        if (!dir.exists(outdir)) stop("outdir must exist: ", outdir)
    }
    
    # Cleanup extracted frames unless requested
    if (!isTRUE(keep_frames)) {
        on.exit(unlink(outdir, recursive = TRUE, force = TRUE), add = TRUE)
    }
    
    # Extract frames (sampling if fps_extract provided)
    frames <- ovideo::ov_video_frames(
        video_file = video_file,
        start_time = start_time,
        end_time   = end_time,
        outdir     = outdir,
        fps        = fps_extract,
        format     = format,
        jpg_quality = jpg_quality,
        debug      = debug
    )
    
    if (!length(frames)) {
        return(list(
            detections = data.frame(),
            frames = character(),
            meta = list(video_file = video_file, start_time = start_time, end_time = end_time,
                        fps_extract = fps_extract, outdir = outdir)
        ))
    }
    
    # Run detection on extracted frames (regular image path workflow)
    dets <- ovml_yolo_detect(
        net = net,
        image_file = frames,
        conf = conf,
        nms_conf = nms_conf,
        classes = if (missing(classes)) NULL else classes,
        batch_size = batch_size
    )
    
    # Add frame_number and timestamps
    # image_number from ovml_yolo_detect is 1-based across the frames vector. :contentReference[oaicite:3]{index=3}
    dets$frame_number <- dets$image_number
    
    # If we extracted at fps_extract, timestamps align with that sampling grid.
    # If fps_extract is NULL (all frames), we need the true fps to compute time.
    if (is.null(fps_extract)) {
        if (missing(fps_video)) {
            vi <- av::av_video_info(video_file)
            fps_video <- vi$video$framerate
        }
        dt <- 1 / fps_video
    } else {
        dt <- 1 / fps_extract
    }
    
    dets$time_s <- start_time + (dets$frame_number - 1) * dt
    
    list(
        detections = dets,
        frames = frames,
        meta = list(
            video_file = video_file,
            start_time = start_time,
            end_time = end_time,
            fps_extract = fps_extract,
            fps_video = if (!missing(fps_video)) fps_video else NA_real_,
            outdir = outdir
        )
    )
}
