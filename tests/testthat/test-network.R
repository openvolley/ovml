context("overall network tests")

dn <- ovml_yolo(version = "4-tiny")

test_that("basic inference", {
    img <- ovml_example_image()
    res <- ovml_yolo_detect(dn, img, conf = 0.6)
    expect_true(setequal(res$class, c("person")))
    res <- ovml_yolo_detect(dn, img, conf = 0.1)
    expect_true(setequal(res$class, c("person", "backpack")))
    res <- ovml_yolo_detect(dn, img, conf = 0.05)
    expect_true(setequal(res$class, c("backpack", "bench", "chair", "handbag", "person", "traffic light")))
})

test_that("batch inference", {
    img <- ovml_example_image(c(1, 2))
    res <- ovml_yolo_detect(dn, img)
    expect_true(setequal(res$image_number, 1:2))
    expect_true(setequal(res$class, c("bicycle", "car", "dog", "person", "truck")))

    ## do the same image twice, expect the same results for each
    img <- rep(ovml_example_image(), 2)
    res <- ovml_yolo_detect(dn, img)
    chk1 <- res[res$image_number == 1, ]
    chk2 <- res[res$image_number == 2, ]
    rownames(chk2) <- NULL
    expect_equal(chk1[, -1], chk2[, -1])
})
