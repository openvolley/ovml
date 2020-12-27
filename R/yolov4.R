yolo4_mish <- nn_module("mish",
                        initialize = function() {},
                        forward = function(x) {
                            x * torch_tanh(nnf_softplus(x))
                        })

## for debugging
##conv_trace <- nn_module("conv_trace",
##                        initialize = function(in_channels, out_channels, kernel_size, stride = 1, padding = 0, groups = 1, bias = TRUE) {
##                            self$conv <- nn_conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = bias)
##                        },
##                        forward = function(x) {
##                            cat("conv_trace\n")
##                            cat("input dim: "); print(dim(x))
##                            print(self$conv)
##                            self$conv(x)
##                        })

yolo4_create_darknet_modules <- function(blocks, device) {
    net <- list()
    out_channels <- out_widths <- out_heights <- c()
    index <- 1L ## 1-based index into net
    for (i in seq_along(blocks)) {
        block <- blocks[[i]]
        module <- nn_sequential()
        block_type <- block$type
        if (block_type == "net") {
            out_channels <- c(out_channels, get_int_from_cfg(block, "channels", 3))
            out_widths <- c(out_widths, get_int_from_cfg(block, "width", 0))
            out_heights <- c(out_heights, get_int_from_cfg(block, "height", 0))
            next ## don't increment index
        } else if (block_type == "convolutional") {
            activation <- get_string_from_cfg(block, "activation")
            batch_normalize <- get_int_from_cfg(block, "batch_normalize")
            filters <- get_int_from_cfg(block, "filters")
            kernel_size <- get_int_from_cfg(block, "size")
            stride <- get_int_from_cfg(block, "stride", 1)
            pad <- ifelse(get_int_from_cfg(block, "pad") > 0, (kernel_size-1)/2, 0)
            ##module$add_module(paste0("conv_", index), conv_trace(in_channels = tail(out_channels, 1), out_channels = filters, kernel_size = kernel_size, stride = stride, padding = pad, groups = 1, bias = batch_normalize < 1))
            module$add_module(paste0("conv_", index), nn_conv2d(in_channels = tail(out_channels, 1), out_channels = filters, kernel_size = kernel_size, stride = stride, padding = pad, groups = 1, bias = batch_normalize < 1))
            width <- (tail(out_widths, 1) + 2 * pad - kernel_size )/stride + 1
            height <- (tail(out_heights, 1) + 2 * pad - kernel_size )/stride + 1
            ##printf("%5zu %-6s %4d    %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d\n", i, "conv", filters, kernel_size, kernel_size, stride, out_widths.back(), out_heights.back(), out_channels.back(), width, height, filters);
            out_heights <- c(out_heights, height)
            out_widths <- c(out_widths, width)
            out_channels <- c(out_channels, filters)
            if (batch_normalize > 0) {
                bn <- nn_batch_norm2d(filters, affine = TRUE, track_running_stats = TRUE)
                module$add_module(paste0("batch_norm_", index), bn)
            }
            if (activation == "leaky") {
                module$add_module(paste0("leaky_", index), nn_leaky_relu(0.1, inplace = TRUE))
            } else if (activation == "mish") {
                module$add_module(paste0("mish_", index), yolo4_mish())
            } else if (activation == "linear") {
                ## do nothing
            } else {
                stop("unsupported activation: ", activation)
            }
        } else if (block_type == "upsample") {
            stride <- get_int_from_cfg(block, "stride", 1)
            width <- tail(out_widths, 1) * stride
            height <- tail(out_heights, 1) * stride
            module$add_module(paste0("upsample_", index), yolo3_upsample_layer(stride))
            out_widths <- c(out_widths, width)
            out_heights <- c(out_heights, height)
            out_channels <- c(out_channels, tail(out_channels, 1))
        } else if (block_type == "maxpool") {
            stride <- get_int_from_cfg(block, "stride", 1)
            size <- get_int_from_cfg(block, "size", 1)
            module$add_module(paste0("maxpool_", index), yolo3_maxpool_layer_2d(size, stride))
            width <- tail(out_widths, 1) / stride
            height <- tail(out_heights, 1) / stride
            out_widths <- c(out_widths, width)
            out_heights <- c(out_heights, height)
            out_channels <- c(out_channels, tail(out_channels, 1))
        } else if (block_type == "shortcut") {
            ## skip connection
            block$from <- get_int_from_cfg(block, "from", 0) + index
            ## from values are always negative, adding +index makes them indexed as per modules in the $net
            blocks[[i]] <- block
            ## placeholder
            module$add_module(paste0("shortcut_", index), yolo3_empty_layer())
            out_channels <- c(out_channels, tail(out_channels, 1))
            out_heights <- c(out_heights, tail(out_heights, 1))
            out_widths <- c(out_widths, tail(out_widths, 1))
        } else if (block_type == "route") {
            layers_info <- get_string_from_cfg(block, "layers", "")
            layers <- as.integer(strsplit(layers_info, ",")[[1]])
            groups <- get_int_from_cfg(block, "groups", 0)
            total_channel <- 0L

            for (j in seq_along(layers)) {
                ix <- layers[j] + if (layers[j] > 0) 1L else index ##int ix = layers[j] > 0 ? layers[j] : (i + layers[j]);
                ## positive values use 0-based indexing in the cfg file, but we are using 1-based indexing here, hence the +1L
                total_channel <- total_channel + out_channels[ix + 1L] ##total_channel += out_channels[ ix ];
                ## +1 to index into out_channels because the first is from the net module
                layers[j] <- ix
            }
            block$layers <- layers
            if (length(layers) == 4) {
                ##printf("%5zu %-6s %d %d %d %d                              %d\n", i, "route",  layers[0], layers[1], layers[2], layers[3], total_channel);
            } else if (length(layers) == 2) {
                ##printf("%5zu %-6s %d %d                                    %d\n", i, "route", layers[0], layers[1], total_channel);
            } else if (length(layers) == 1) {
                if (groups != 0) {
                    total_channel <- total_channel / groups
                    block$chunk_size <- total_channel
                }
                ##printf("%5zu %-6s %d                                       %d\n", i, "route",  layers[0], total_channel);
            } else {
                stop("route not supported")
            }
            out_channels <- c(out_channels, total_channel)
            out_heights <- c(out_heights, tail(out_heights, 1))
            out_widths <- c(out_widths, tail(out_widths, 1))
            blocks[[i]] <- block
            ## placeholder
            module$add_module(paste0("route_", index), yolo3_empty_layer())
        } else if (block_type == "yolo") {
            mask_info <- get_string_from_cfg(block, "mask", "")
            masks <- as.integer(strsplit(mask_info, ",")[[1]])
            anchor_info <- get_string_from_cfg(block, "anchors", "")
            anchors <- as.integer(strsplit(anchor_info, ",")[[1]])
            if (length(anchors) != 2 * get_int_from_cfg(block, "num")) stop("expecting anchors to be 2-column matrix")
            anchor_points <- matrix(anchors, ncol = 2, byrow = TRUE)[masks+1, ]
            module <- yolo3_detection_layer(anchor_points, device)
            out_channels <- c(out_channels, tail(out_channels, 1))
            out_heights <- c(out_heights, tail(out_heights, 1))
            out_widths <- c(out_widths, tail(out_widths, 1))
        } else {
            stop("unsupported operator: ", block_type)
        }
        module_key <- paste0("layer_", i-1L)
        net[[module_key]] <- module
        index <- index + 1L
        if (length(out_widths) != index || length(out_heights) != index || length(out_channels) != index) {
            stop("something wrong in creating darknet modules")
        }
    }
    net <- nn_module_list(net)
    list(blocks = blocks, net = net)
}

yolo4_darknet <- nn_module("darknet",
                     initialize = function(cfg_file, device) {
                         blocks <- yolo3_read_darknet_cfg(cfg_file)
                         temp <- yolo4_create_darknet_modules(blocks, device) ## create and register modules
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
                                 groups <- get_int_from_cfg(block, "groups", 0)
                                 group_id <- get_int_from_cfg(block, "group_id", 0)
                                 layers <- block$layers ## already processed
                                 if (length(layers) == 1) {
                                     x <- outputs[[layers[1]]]
                                     if (groups != 0) {
                                         chunk_size <- get_int_from_cfg(block, "chunk_size", 0)
                                         group_tensor <- torch_split(x, chunk_size, 1 + dim_off)
                                         x <- group_tensor[[group_id + dim_off]]
                                     }
                                 } else if (length(layers) == 2) {
                                     x <- torch_cat(list(outputs[[layers[1]]], outputs[[layers[2]]]), 1 + dim_off)
                                 } else if (length(layers) == 4) {
                                     x <- torch_cat(list(outputs[[layers[1]]], outputs[[layers[2]]], outputs[[layers[3]]], outputs[[layers[4]]]), 1 + dim_off)
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
                         out <- as.array(result) ## copy to cpu
                         for (i in seq_along(outputs)) outputs[[i]] <- torch::torch_zeros(1)
                         x <- torch::torch_zeros(1)
                         rm(outputs)
                         result <- torch::torch_zeros(1)
                         gc()
                         out
                     })
