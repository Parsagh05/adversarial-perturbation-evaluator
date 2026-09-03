# Few-shot notebooks

Empty until the first few-shot adapter lands. Few-shot models need a normal
reference set the zero-shot ones do not: a k-shot memory bank built from
training images. That memory is static — it is built once from clean normal
images and never updated while the cohort is scored — so it does not break the
matched clean-versus-adversarial protocol, but it does add two settings that
every notebook here must pin and record: `k` and the seed that selects which
normal images are drawn.
