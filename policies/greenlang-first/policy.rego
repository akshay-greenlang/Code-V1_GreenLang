package greenlang_first

import rego.v1

default allow := false

allow if {
	input.product == "greenlang"
}

allow if {
	input.repository == "Code-V1_GreenLang"
}
