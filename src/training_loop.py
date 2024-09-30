def _train(
    training_features: torch.Tensor,
    training_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    model: nn.Module,
    optimizer: AdamOptimizer,
    loss_function: nn.Module,
    epochs: int
):
    training_loss: List[float] = []
    validation_loss: List[float] = []
    training_accuracy: List[float] = []
    validation_accuracy: List[float] = []

    for epoch in range(epochs):
        model.train()

        training_ypred = model(training_features)
        train_loss = loss_function(training_ypred, training_labels)
        training_loss.append(train_loss.item())

        _, predicted = torch.max(training_ypred, 1)
        correct = (predicted == training_labels).sum().item()
        accuracy = correct / training_labels.size(0)
        training_accuracy.append(accuracy)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_ypred = model(validation_features)
            val_loss = loss_function(validation_ypred, validation_labels)
            validation_loss.append(val_loss.item())

            _, val_predicted = torch.max(validation_ypred, 1)
            val_correct = (val_predicted == validation_labels).sum().item()
            val_accuracy = val_correct / validation_labels.size(0)
            validation_accuracy.append(val_accuracy)

        wandb.log({
            "epoch": epoch,
            "training_loss": train_loss.item(),
            "validation_loss": val_loss.item(),
            "training_accuracy": accuracy,
            "validation_accuracy": val_accuracy
        })

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Training Loss = {train_loss.item():.4f} | Validation Loss = {val_loss.item():.4f} | Training Acc = {accuracy:.4f} | Validation Acc = {val_accuracy:.4f}")

    return training_loss, validation_loss, training_accuracy, validation_accuracy