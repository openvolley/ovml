#' Construct YOLO network
#'
#' @references https://github.com/pjreddie/darknet
#' @param version integer or string: currently only 3 (YOLO v3)
#' @param device string: "cpu" or "cuda"
#' @param weights_file string: either the path to the weights file that already exists on your system or "auto". If "auto", the weights file will be downloaded if necessary and stored in the directory given by [ovml_cache_dir]
#'
#' @return A YOLO network object
#'
# @examples
#'
#' @export
ovml_yolo <- function(version = 3, device = "cpu", weights_file = "auto") {
    if (is.numeric(version)) version <- as.character(version)
    assert_that(version %in% c("3"))##, "5s", "5m", "5l", "5x"))
    assert_that(is.string(device))
    device <- tolower(device)
    device <- match.arg(device, c("cpu", "cuda"))
    if (device == "cuda" && !cuda_is_available()) {
        warning("'cuda' device not available, using 'cpu'")
        device <- "cpu"
    }
    if (version == "3") {
        dn <- yolo3_darknet(system.file(paste0("extdata/yolo/yolov", version, ".cfg"), package = "ovml"), device = device)
        if (length(weights_file) && nzchar(weights_file) && !is.na(weights_file)) {
            if (identical(tolower(weights_file), "auto")) {
                weights_file <- ovml_download_if("https://pjreddie.com/media/files/yolov3.weights", dest = paste0("yolov", version, ".weights"))
            }
            if (file.exists(weights_file)) dn$load_weights(weights_file)
        }
        if (device == "cuda") dn$to(torch_device(device))
        dn$eval()
    }
    dn
}

#' Detect objects in image using a YOLO network
#'
#' @param net yolo: as returned by [ovml_yolo]
#' @param image_file character: path to one or more image files
#' @param conf scalar: confidence level
#' @param nms_conf scalar: non-max suppression confidence level
#' @param num_classes integer: number of classes that the network was trained on
#' @param input_image_size integer: image size expected by the network
#' @param class_labels character: the labels corresponding to the class numbers
#'
#' @return A data.frame with columns "image_number", "class", "score", "xmin", "xmax", "ymin", "ymax"
#'
#' @seealso [ovml_yolo]
#'
#' @examples
#' \dontrun{
#'   dn <- ovml_yolo()
#'   img <- system.file("extdata/images/2019_03_01-KATS-BEDS-frame.jpg", package = "ovml")
#'   res <- ovml_yolo_detect(dn, img)
#'   ovml_plot(img, res)
#' }
#' @export
ovml_yolo_detect <- function(net, image_file, conf = 0.6, nms_conf = 0.4, num_classes = 80, input_image_size = 416L, class_labels = ovml_class_labels()) {
    imgs <- lapply(image_file, function(im) {
        image <- image_read(im) ## h x w x rgb
        resized_image <- as.numeric(image_data(image_resz(image, input_image_size, preserve_aspect = YOLO3_LETTERBOXING), "rgb"))
        list(tensor = torch_tensor(aperm(array(resized_image, dim = c(1, dim(resized_image))), c(1, 4, 2, 3))), original_wh = image_wh(image))
    })
    img_tensor <- torch_cat(lapply(imgs, function(z) z$tensor), dim = 1)
    output <- as.array(net$forward(img_tensor))
    owh <- do.call(rbind, lapply(imgs, function(z) z$original_wh))
    write_results(output, num_classes = num_classes, confidence = conf, nms_conf = nms_conf, original_wh = owh, input_image_size = input_image_size, class_labels = class_labels)
}

#' Class labels
#'
#' @param dataset string: which dataset?
#'
#' @return A character vector of class labels
#'
#' @export
ovml_class_labels <- function(dataset = "coco") {
    assert_that(is.string(dataset))
    dataset <- tolower(dataset)
    dataset <- match.arg(dataset, "coco")
    ## old, with missing elements. This might be needed with the original C darknet implementation?
    ##c("person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", NA_character_, "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", NA_character_, "backpack", "umbrella", NA_character_, NA_character_, "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", NA_character_, "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", NA_character_, "diningtable", NA_character_, NA_character_, "toilet", NA_character_, "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", NA_character_, "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
    c("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
}
