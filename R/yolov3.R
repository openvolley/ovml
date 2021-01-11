dim_off <- 1L ## 0-based dimension indexing in C++/Python, but 1-based dimension indexing here
YOLO_LETTERBOXING <- TRUE

yolo3_empty_layer <- nn_module("empty_layer",
                        initialize = function() {},
                        forward = function(x) x)

yolo3_trace_layer <- nn_module("trace_layer",
                        initialize = function(id) {self$.id <- id},
                        forward = function(x) {
                            cat(self$.id, ": dim x = ", dim(x), "\n")
                            x
                        })

yolo3_upsample_layer <- nn_module("upsample_layer",
                           initialize = function(stride) {
                               self$.stride <- stride
                           },
                           forward = function(x) {
                               if (length(dim(x)) == 4) {
                                   w <- dim(x)[3] * self$.stride
                                   h <- dim(x)[4] * self$.stride
                                   x <- nnf_interpolate(x, c(w, h), mode = "nearest")
                               } else if (length(dim(x)) == 3) {
                                   w <- dim(x)[3] * self$.stride
                                   x <- nnf_interpolate(x, w, mode = "nearest")
                               } else {
                                   stop("expecting 3- or 4-D input to yolo3_upsample_layer")
                               }
                               x
                           }
                           )

yolo3_maxpool_layer_2d <- nn_module("maxpool_layer_2d",
                            initialize = function(kernel_size, stride) {
                                self$.kernel_size <- kernel_size
                                self$.stride <- stride
                            },
                            forward = function(x) {
                                if (self$.stride != 1) {
                                    nn_max_pool2d(c(self$.kernel_size, self$.kernel_size), stride = c(self$.stride, self$.stride))(x)
                                } else {
                                    pad <- self$.kernel_size - 1
                                    padded_x <- nnf_pad(x, pad = c(0, pad, 0, pad), mode = "replicate")
                                    nn_max_pool2d(c(self$.kernel_size, self$.kernel_size), stride = c(self$.stride, self$.stride))(padded_x)
                                }
                            }
                            )

## anchors should be n x 2 matrix
yolo3_detection_layer <- nn_module("detection_layer",
                            initialize = function(anchors, device) {
                                self$.anchors <- anchors
                                self$.device <- device
                            },
                            forward = function(prediction, inp_dim, num_classes, device) {
                                batch_size <- dim(prediction)[1]
                                stride <- floor(inp_dim / dim(prediction)[3])
                                grid_size <- floor(inp_dim / stride)
                                bbox_attrs <- 5 + num_classes
                                num_anchors <- nrow(self$.anchors)
                                anchors <- self$.anchors/stride
                                result <- prediction$view(c(batch_size, bbox_attrs * num_anchors, grid_size * grid_size))
                                result <- result$transpose(1 + dim_off, 2 + dim_off)$contiguous()
                                result <- result$view(c(batch_size, grid_size * grid_size * num_anchors, bbox_attrs))
                                result$select(2 + dim_off, 0 + dim_off)$sigmoid_()
                                result$select(2 + dim_off, 1 + dim_off)$sigmoid_()
                                result$select(2 + dim_off, 4 + dim_off)$sigmoid_()
                                grid_len <- torch_arange(start = 0, end = grid_size, device = self$.device)
                                args <- torch_meshgrid(list(grid_len, grid_len))
                                x_offset <- args[[2]]$contiguous()$view(c(-1, 1))$to(device = self$.device)
                                y_offset <- args[[1]]$contiguous()$view(c(-1, 1))$to(device = self$.device)
                                ##if (self$.device == "cuda") {
                                ##    x_offset <- x_offset$cuda()
                                ##    y_offset <- y_offset$cuda()
                                ##}
                                x_y_offset <- torch_cat(list(x_offset, y_offset), 1 + dim_off)$`repeat`(c(1, num_anchors))$view(c(-1, 2))$unsqueeze(0 + dim_off)
                                result$slice(2 + dim_off, 0, 2)$add_(x_y_offset)
                                anchors_tensor <- torch_tensor(anchors, device = self$.device)
                                anchors_tensor <- anchors_tensor$`repeat`(c(grid_size * grid_size, 1))$unsqueeze(0 + dim_off)
                                result$slice(2 + dim_off, 2, 4)$exp_()$mul_(anchors_tensor)
                                result$slice(2 + dim_off, 5, 5 + num_classes)$sigmoid_()
                                result$slice(2 + dim_off, 0, 4)$mul_(stride)
                                result
                            }
                            )

get_string_from_cfg <- function(block, key, default_value = "") {
    if (key %in% names(block)) block[[key]] else default_value
}
get_int_from_cfg <- function(block, key, default_value = 0L) {
    as.integer(get_string_from_cfg(block, key, default_value = default_value))
}

yolo3_read_darknet_cfg <- function(cfg_file) {
    cfg <- readLines(cfg_file)
    blocks <- list()
    block <- list()
    for (line in cfg) {
        line <- str_trim(line)
        if (!nzchar(line) || substr(line, 1, 1) == "#") {
            ## empty line or comment
        } else if (substr(line, 1, 1) == "[") {
            if (length(block)) blocks[[length(blocks)+1]] <- block
            block <- list(type = substr(line, 2, nchar(line)-1))
            if (!block$type %in% c("convolutional", "yolo", "route", "shortcut", "net", "upsample", "maxpool")) warning("block type: ", block$type, " is unknown")
        } else {
            this <- str_trim(strsplit(line, "=")[[1]])
            block <- c(block, setNames(list(this[2]), this[1]))
        }
    }
    if (length(block)) blocks[[length(blocks)+1]] <- block
    blocks
}

yolo3_create_darknet_modules <- function(blocks, device) {
    prev_filters <- 3L
    output_filters <- integer()
    net <- list()
    index <- 1L ## 1-based index into output_filters and net
    filters <- 0L
    for (i in seq_along(blocks)) {
        block <- blocks[[i]]
        module <- nn_sequential()
        layer_type <- block$type
        if (layer_type == "net") {
            next ## don't increment index
        } else if (layer_type == "convolutional") {
                activation <- get_string_from_cfg(block, "activation")
                batch_normalize <- get_int_from_cfg(block, "batch_normalize")
                filters <- get_int_from_cfg(block, "filters")
                kernel_size <- get_int_from_cfg(block, "size")
                stride <- get_int_from_cfg(block, "stride", 1)
                pad <- ifelse(get_int_from_cfg(block, "pad") > 0, (kernel_size-1)/2, 0)
                module$add_module(paste0("conv_", index), nn_conv2d(in_channels = prev_filters, out_channels = filters, kernel_size = kernel_size, stride = stride, padding = pad, groups = 1, bias = batch_normalize < 1))
                if (batch_normalize > 0) {
                    bn <- nn_batch_norm2d(filters, affine = TRUE, track_running_stats = TRUE)
                    module$add_module(paste0("batch_norm_", index), bn)
                }
                if (activation == "leaky") {
                    module$add_module(paste0("leaky_", index), nn_leaky_relu(0.1, inplace = TRUE))
                }
                ##module$add_module(paste0("trace_", index), yolo3_trace_layer(paste0(index, " out")))
            } else if (layer_type == "upsample") {
                stride <- get_int_from_cfg(block, "stride", 1)
                module$add_module(paste0("upsample_", index), yolo3_upsample_layer(stride))
            } else if (layer_type == "maxpool") {
                stride <- get_int_from_cfg(block, "stride", 1)
                size <- get_int_from_cfg(block, "size", 1)
                module$add_module(paste0("maxpool_", index), yolo3_maxpool_layer_2d(size, stride))
            } else if (layer_type == "shortcut") {
                ## skip connection
                block$from <- get_int_from_cfg(block, "from", 0) + index
                ## from values are always negative, adding +index makes them indexed as per modules in the $net
                blocks[[i]] <- block
                ## placeholder
                module$add_module(paste0("shortcut_", index), yolo3_empty_layer())
            } else if (layer_type == "route") {
                layers_info <- get_string_from_cfg(block, "layers", "")
                layers <- as.integer(strsplit(layers_info, ",")[[1]])
                start <- layers[1] ## start is always negative
                if (start < 0) start <- start + index ## now 1-indexed as per index
                end <- if (length(layers) > 1) layers[2] else NA_integer_
                end <- end + 1L ## end values use 0-based indexing in the model cfg, but we use 1-based indexing here
                block$start <- start
                block$end <- end
                blocks[[i]] <- block
                ## placeholder
                module$add_module(paste0("route_", index), yolo3_empty_layer())
                filters <- output_filters[start]
                if (!is.na(end)) filters <- filters + output_filters[end]
            } else if (layer_type == "yolo") {
                mask_info <- get_string_from_cfg(block, "mask", "")
                masks <- as.integer(strsplit(mask_info, ",")[[1]])
                anchor_info <- get_string_from_cfg(block, "anchors", "")
                anchors <- as.integer(strsplit(anchor_info, ",")[[1]])
                anchor_points <- matrix(anchors, ncol = 2, byrow = TRUE)[masks+1, ]
                module <- yolo3_detection_layer(anchor_points, device)
            } else {
                stop("unsupported operator: ", layer_type)
            }
            prev_filters <- filters
            output_filters <- c(output_filters, filters)
            module_key <- paste0("layer_", i-1L)
            net[[module_key]] <- module
            index <- index + 1L
        }
        net <- nn_module_list(net)
        list(blocks = blocks, net = net)
    }


yolo3_load_weights <- function(net, blocks, weight_file) {
    if (!file.exists(weight_file)) stop("weight file does not exist")
    fs <- file(weight_file, "rb")
    ## header info: 5 * int32_t  (3 x (int32) version info: major, minor, revision; then 1 x  (int64) number of images seen during training )
    header_size <- 4L*5L
    ## skip header
    seek(fs, header_size)
    weights <- torch_tensor(readBin(fs, "numeric", n = ceiling(file.size(weight_file)/4)+10, size = 4))
    close(fs)
    nw <- dim(weights)
    index_weight <- 0L ## zero-based tensor indexing
    for (i in seq_along(net)) {
        module_info <- blocks[[i + 1]]
        ## only conv layer need to load weight
        if (module_info$type != "convolutional") next
        conv_imp <- net[[i]][[1]]
        if (inherits(conv_imp, "conv_trace")) conv_imp <- conv_imp$conv
        batch_normalize <- get_int_from_cfg(module_info, "batch_normalize", 0)
        if (batch_normalize > 0) {
            ## second module
            bn_imp <- net[[i]][[2]]
            num_bn_biases <- bn_imp$bias$numel()
            bn_bias <- weights$slice(0 + dim_off, index_weight, index_weight + num_bn_biases)
            index_weight <- index_weight + num_bn_biases
            bn_weights <- weights$slice(0 + dim_off, index_weight, index_weight + num_bn_biases)
            index_weight <- index_weight + num_bn_biases
            bn_running_mean <- weights$slice(0 + dim_off, index_weight, index_weight + num_bn_biases)
            index_weight <- index_weight + num_bn_biases
            bn_running_var <- weights$slice(0 + dim_off, index_weight, index_weight + num_bn_biases)
            index_weight <- index_weight + num_bn_biases
            bn_bias <- bn_bias$view_as(bn_imp$bias)
            bn_weights <- bn_weights$view_as(bn_imp$weight)
            bn_running_mean <- bn_running_mean$view_as(bn_imp$running_mean)
            bn_running_var <- bn_running_var$view_as(bn_imp$running_var)
            bn_imp$bias$set_data(bn_bias)
            bn_imp$weight$set_data(bn_weights)
            bn_imp$running_mean$set_data(bn_running_mean)
            bn_imp$running_var$set_data(bn_running_var)
        } else {
            num_conv_biases <- conv_imp$bias$numel()
            conv_bias <- weights$slice(0 + dim_off, index_weight, index_weight + num_conv_biases)
            index_weight <- index_weight + num_conv_biases
            conv_bias <- conv_bias$view_as(conv_imp$bias)
            conv_imp$bias$set_data(conv_bias)
        }
        num_weights <- conv_imp$weight$numel()
        conv_weights <- weights$slice(0 + dim_off, index_weight, index_weight + num_weights)
        index_weight <- index_weight + num_weights
        conv_weights <- conv_weights$view_as(conv_imp$weight)
        conv_imp$weight$set_data(conv_weights)
    }
    if (index_weight != nw) {
        warning("finished reading weights at ", index_weight, " of ", nw, " weights, something has gone wrong")
    }
}

yolo3_darknet <- nn_module("darknet",
                     initialize = function(cfg_file, device) {
                         if (!file.exists(cfg_file)) stop("config file does not exist")
                         blocks <- yolo3_read_darknet_cfg(cfg_file)
                         temp <- yolo3_create_darknet_modules(blocks, device) ## create and register modules
                         self$blocks <- temp$blocks
                         self$net <- temp$net
                         self$device <- device
                         temp <- unique(as.integer(unlist(lapply(temp$blocks, function(z) if (z$type == "yolo") z$classes))))
                         if (length(temp) == 1) {
                             self$num_classes <- temp
                         } else {
                             stop("inconsistent number of classes in the model cfg file")
                         }
                     },
                     load_weights = function(weight_file) {
                         self$net <- yolo3_load_weights(self$net, self$blocks, weight_file)
                     },
                     forward = function(x) {
                         outputs <- list() ## will be indexed as for net
                         result <- NULL
                         for (i in seq_along(self$net)) {
                             block <- self$blocks[[i+1]]
                             layer_type <- block$type
                             if (layer_type == "net") next
                             if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool") {
                                 x <- self$net[[i]]$forward(x)
                                 outputs[[i]] <- x
                             } else if (layer_type == "route") {
                                 start <- block$start ## block$start and $end are indexed as per i here
                                 end <- block$end
                                 if (is.na(end)) {
                                     x <- outputs[[start]]
                                 } else {
                                     x <- torch_cat(list(outputs[[start]], outputs[[end]]), 1 + dim_off)
                                 }
                                 outputs[[i]] <- x
                             } else if (layer_type == "shortcut") {
                                 x <- outputs[[i-1]] + outputs[[block$from]]
                                 outputs[[i]] <- x
                             } else if (layer_type == "yolo") {
                                 net_info <- self$blocks[[1]]
                                 inp_dim <- get_int_from_cfg(net_info, "height", 0)
                                 num_classes <- get_int_from_cfg(block, "classes", 0)
                                 x <- self$net[[i]]$forward(x, inp_dim, num_classes, self$device)
                                 if (is.null(result)) {
                                     result <- x
                                 } else {
                                     result <- torch_cat(list(result, x), 1 + dim_off)
                                 }
                                 outputs[[i]] <- x
                             }
                         }
                         ## clean some stuff up, R doesn't yet appear to properly release the memory used by libtorch when no longer needed?
                         out <- as.array(result$to(device = torch_device("cpu"))) ## copy to cpu
                         for (i in seq_along(outputs)) outputs[[i]] <- torch::torch_zeros(1, device = self$device)
                         x <- torch::torch_zeros(1, device = self$device)
                         rm(outputs)
                         result <- torch::torch_zeros(1, device = self$device)
                         gc()
                         out
                     }
                     )


## the iou of two bounding boxes
bbox_iou <- function(box1, box2) {
    ## intersection rectangle
    inter_rect_x1 <- pmax(box1[1], box2[, 1])
    inter_rect_y1 <- pmax(box1[2], box2[, 2])
    inter_rect_x2 <- pmin(box1[3], box2[, 3])
    inter_rect_y2 <- pmin(box1[4], box2[, 4])
    ## intersection area
    inter_area <- pmax(inter_rect_x2 - inter_rect_x1 + 1, 0) * pmax(inter_rect_y2 - inter_rect_y1 + 1, 0)
    ## box areas
    b1_area = (box1[3] - box1[1] + 1)*(box1[4] - box1[2] + 1)
    b2_area = (box2[, 3] - box2[, 1] + 1)*(box2[, 4] - box2[, 2] + 1)
    ## iou
    inter_area / (b1_area + b2_area - inter_area)
}

## apply nms and convert results matrix
## original_wh should be an n x 2 matrix for n images
write_results <- function(prediction, num_classes, confidence = 0.6, nms_conf = 0.4, original_wh, input_image_size, class_labels) {
    if (is.null(dim(original_wh))) original_wh <- matrix(original_wh, ncol = 2, byrow = TRUE)
    if (nrow(original_wh) != dim(prediction)[1]) {
        stop("number of images in prediction tensor does not match the number of rows in original_wh")
    }
    if (missing(num_classes)) num_classes <- dim(prediction)[3] - 5
    mask_idx <- prediction[, , 5] > confidence
    if (!any(mask_idx)) return(data.frame(class = character(), score = numeric(), xmin = numeric(), xmax = numeric(), ymin = numeric(), ymax = numeric(), stringsAsFactors = FALSE))
    ## coords in predictions are xmid, ymid, width, height
    box_corner <- array(dim = c(dim(prediction)[1:2], 4))
    box_corner[, , 1] <- prediction[, , 1] - prediction[, , 3]/2 ## xmin
    box_corner[, , 2] <- prediction[, , 2] - prediction[, , 4]/2 ## ymin
    box_corner[, , 3] <- prediction[, , 1] + prediction[, , 3]/2 ## xmax
    box_corner[, , 4] <- prediction[, , 2] + prediction[, , 4]/2 ## ymax
    prediction[, , 1:4] <- box_corner
    batch_size <- dim(prediction)[1] ## number of images in batch
    output <- matrix(nrow = 0, ncol = 8)
    for (ind in seq_len(batch_size)) {
        image_pred <- prediction[ind, , ]
        ##confidence thresholding
        ##NMS
        max_conf <- apply(image_pred[, 6:(num_classes + 5), drop = FALSE], 1, max)
        max_conf_score <- apply(image_pred[, 6:(num_classes + 5), drop = FALSE], 1, which.max) ## indices
        image_pred <- cbind(image_pred[, 1:5, drop = FALSE], max_conf, max_conf_score)
        image_pred_ <- image_pred[image_pred[, 5] > confidence, , drop = FALSE]
        if (nrow(image_pred_) < 1) next
        ## Get the various classes detected in the image
        img_classes <- unique(image_pred_[, 7])
        for (cls in img_classes) {
            ## perform NMS
            ## get the detections with one particular class
            image_pred_class <- image_pred_[image_pred_[, 7] == cls, , drop = FALSE]
            ##sort the detections such that the entry with the maximum objectness
            ##confidence is at the top
            image_pred_class <- image_pred_class[order(image_pred_class[, 5], decreasing = TRUE), , drop = FALSE]
            ndets <- nrow(image_pred_class)   ## Number of detections
            for (i in seq_len(ndets-1)) {
                if (i >= nrow(image_pred_class)) break
                ## Get the IOUs of all boxes that come after the one we are looking at in the loop
                ijdx <- seq(from = i+1, to = nrow(image_pred_class), by = 1)
                ious <- tryCatch(bbox_iou(image_pred_class[i, , drop = FALSE], image_pred_class[ijdx, , drop = FALSE]), error = function(e) NULL)
                if (is.null(ious)) break
                ## Zero out all the detections that have IoU > threshhold
                image_pred_class[ijdx[ious >= nms_conf], 5] <- 0
                ## Remove the non-zero entries
                image_pred_class <- image_pred_class[image_pred_class[, 5] > 0, , drop = FALSE]
            }
            output <- rbind(output, cbind(ind, image_pred_class))
        }
    }
    colnames(output) <- NULL
    class_num <- output[, 8]
    if (!missing(class_labels) && length(class_labels)) {
        class_label <- rep(NA_character_, length(class_num))
        tempidx <- which(class_num %in% seq_along(class_labels))
        class_label[tempidx] <- class_labels[class_num[tempidx]]
    } else {
        class_label <- as.character(class_num)
    }
    oh <- original_wh[output[, 1], 2] ## one per output box
    bb <- rescale_boxes(output[, 2:5, drop = FALSE], original_w = original_wh[output[, 1], 1], original_h = oh, input_image_size = input_image_size, letterboxing = YOLO_LETTERBOXING)
    ## testing
    ##bb <- output[, 2:5]; oh <- 416L
    data.frame(image_number = output[, 1], class = class_label, score = output[, 7],
               xmin = bb[, 1] + 1, xmax = bb[, 3] + 1,
               ymin = oh + 1 - bb[, 4],
               ymax = oh + 1 - bb[, 2], stringsAsFactors = FALSE)
}

rescale_boxes <- function(bboxes, original_w, original_h, input_image_size, letterboxing = TRUE) {
    ## raw predicted bboxes are 416 x 416, so we need to scale them to the aspect-ratio image
    iwh <- cbind(original_w, original_h)
    if (letterboxing) {
        iwh <- t(apply(iwh, 1, function(z) z/max(z)))
        sc <- iwh * input_image_size ## the letterbox size, in pixels
        sco <- (1-iwh) / 2 * input_image_size ## the letterbox margins, in pixels
        bboxes[, c(1, 3)] <- (bboxes[, c(1, 3)] - sco[, 1]) / sc[, 1] * original_w
        bboxes[, c(2, 4)] <- (bboxes[, c(2, 4)] - sco[, 2]) / sc[, 2] * original_h
    } else {
        ## simple scaling from 416 x 416 to original w x h
        bboxes[, c(1, 3)] <- bboxes[, c(1, 3)] / input_image_size * original_w
        bboxes[, c(2, 4)] <- bboxes[, c(2, 4)] / input_image_size * original_h
    }
    bboxes
}

