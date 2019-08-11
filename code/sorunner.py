from deepobs.pytorch.runners import PTRunner
import numpy as np


class SORunner(PTRunner):
    """A second order runner. Can run a normal training loop with grad-thing enabled with fixed
    hyperparams.
    Methods:
        training: Performs the training on a testproblem instance.
    """

    def __init__(self, optimizer_class, hyperparameter_names):
        super(SORunner, self).__init__(optimizer_class, hyperparameter_names)

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    'Not possible to use tensorboard for pytorch. Reason: ' + e, ImportWarning)
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):

            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(
                epoch_count, num_epochs))

            loss_, acc_ = self.evaluate(tproblem, phase='TRAIN')
            train_losses.append(loss_)
            train_accuracies.append(acc_)

            loss_, acc_ = self.evaluate(tproblem, phase='VALID')
            valid_losses.append(loss_)
            valid_accuracies.append(acc_)

            loss_, acc_ = self.evaluate(tproblem, phase='TEST')
            test_losses.append(loss_)
            test_accuracies.append(acc_)

            print("********************************")

            # estimate a new Hessian, drop the learning rate
            if epoch_count > 0:
                opt.start_estimate(lr = None)

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()
                    batch_loss.backward(create_graph=True)
                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print("Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                epoch_count, batch_count, batch_loss))
                        if tb_log:
                            summary_writer.add_scalar(
                                'loss', batch_loss.item(), global_step)

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                train_losses, \
                    valid_losses, \
                    test_losses, \
                    train_accuracies, \
                    valid_accuracies, \
                    test_accuracies, \
                    minibatch_train_losses = self._abort_routine(
                        epoch_count,
                        num_epochs,
                        train_losses,
                        valid_losses,
                        test_losses,
                        train_accuracies,
                        valid_accuracies,
                        test_accuracies,
                        minibatch_train_losses)
                break
            else:
                continue

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            'valid_losses': valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            'valid_accuracies': valid_accuracies,
            "test_accuracies": test_accuracies
        }

        return output
