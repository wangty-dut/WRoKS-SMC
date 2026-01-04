def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood,  # Define log_values function to record and print various metrics during training
               reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()  # Calculate average loss
    grad_norms, grad_norms_clipped = grad_norms  # Get gradient norms and clipped gradient norms

    print('\nepoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))  # Print current epoch, batch and average loss

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))  # Print gradient norms and clipped gradient norms

    if not opts.no_tensorboard:  # If tensorboard logging is enabled
        tb_logger.log_value('avg_cost', avg_cost, step)  # Record average loss

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)  # Record actor loss
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)  # Record negative log likelihood

        tb_logger.log_value('grad_norm', grad_norms[0], step)  # Record gradient norm
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)  # Record clipped gradient norm

        if opts.baseline == 'critic':  # If baseline model is critic
            tb_logger.log_value('critic_loss', bl_loss.item(), step)  # Record critic loss
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)  # Record critic gradient norm
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)  # Record clipped critic gradient norm


def log_values_sl(cost, grad_norms, epoch, batch_id,  # Define log_values_sl function to record and print various metrics during training
                  step, loss, tb_logger, opts):
    avg_cost = cost.mean().item()  # Calculate average loss
    grad_norms, grad_norms_clipped = grad_norms  # Get gradient norms and clipped gradient norms

    # Print current epoch, batch, loss and average loss
    print('\nepoch: {}, train_batch_id: {}, loss: {}, avg_cost: {}'.format(epoch, batch_id, loss,
                                                                           avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))  # Print gradient norms and clipped gradient norms

    # Record values to tensorboard
    if not opts.no_tensorboard:  # If tensorboard logging is enabled
        tb_logger.log_value('avg_cost', avg_cost, step)  # Record average loss

        tb_logger.log_value('actor_loss', loss.mean().item(), step)  # Record training loss

        tb_logger.log_value('grad_norm', grad_norms[0], step)  # Record gradient norm

        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)  # Record clipped gradient norm