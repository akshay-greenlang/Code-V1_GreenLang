package greenlang_first

import rego.v1

test_greenlang_product_allowed if {
	allow with input as {"product": "greenlang"}
}

test_other_product_denied if {
	not allow with input as {"product": "other"}
}
