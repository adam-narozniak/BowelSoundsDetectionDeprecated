import flwr as fl

if __name__ == "__main__":
    strategy_config = dict(fraction_fit=1.,
                           fraction_evaluate=1.,
                           min_fit_clients=2,
                           min_evaluate_clients=2,
                           min_available_clients=2)
    strategy = fl.server.strategy.FedAvg(**strategy_config)
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy
    )
