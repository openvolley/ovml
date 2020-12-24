context("overall network tests")

test_that("basic inference", {
    dn <- ovml_yolo()
    img <- ovml_example_image()
    res <- ovml_yolo_detect(dn, img, conf = 0.6)
    expect_true(setequal(res$class, c("person")))
    res <- ovml_yolo_detect(dn, img, conf = 0.3)
    expect_true(setequal(res$class, c("bench", "clock", "person", "tennis racket")))
    res <- ovml_yolo_detect(dn, img, conf = 0.1)
    expect_true(setequal(res$class, c("backpack", "bench", "clock", "person", "sports ball", "tennis racket")))
})

test_that("batch inference", {
    dn <- ovml_yolo()
    img <- ovml_example_image(c(1, 2))
    res <- ovml_yolo_detect(dn, img)
    expect_true(setequal(res$image_number, 1:2))
    expect_true(setequal(res$class, c("bicycle", "dog", "person", "truck")))
})
