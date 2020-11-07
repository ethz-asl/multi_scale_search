s_l1 expects the x_a node number or for pickup actions the "x_a x_item" node numbers.
The policy for the pickup action where the item is not present at x_a takes the value "x_a none"

a_l1 is either a nav action, "pickup" or "release"

the policy includes the start node in the layer below and then a series of actions. Different start nodes are seperated by a dot (.)
If a_l1 = pickup the policy start node requires again the x_a and x_item start node. If the item is not present x_a suffices.


