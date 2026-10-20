# tflint settings for every module under infra/terraform (issue #23):
# tflint --chdir=infra/terraform --recursive --config="$PWD/infra/terraform/.tflint.hcl"
plugin "terraform" {
  enabled = true
  preset  = "all"
}
