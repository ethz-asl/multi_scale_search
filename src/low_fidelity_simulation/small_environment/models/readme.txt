t_expected files hold the expected runtime for an action/subroutine. 

For the nav actions there is one time for each node. Nodes in which the agent can not start or end up in take the penalty value 1000.

look_around actions differentiate the case where the item is present and where it's not present. 
First come the expected times for the case where the items are not present, then the case where the items are present.

obs_prob files hold the observation probability for an item for an action. 
The first number indicates the node the agent ends up in. Then there is one number for each node which is the observation probability for observing the item if it is in that node. 
