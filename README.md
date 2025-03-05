# Asynchronous Federated Learning
This federated learning version lets clients join in model updates without strict synchronization, even with differing device resources. 

Clients can independently perform computation tasks and send local training results to the central server. The server aggregates global models upon receiving some clients' updates, without waiting for all clients to finish training.

In this version, like the previous one [A-federal-learning-approach-that-allows-clients-to-access-and-disconnect-at-any-time][(https://github.com/2471845937/A-federal-learning-approach-that-allows-clients-to-access-and-disconnect-at-any-time)](https://github.com/2471845937/A-federal-learning-approach-that-allows-clients-to-access-and-disconnect-at-any-time), clients can freely connect to and disconnect from the server.
