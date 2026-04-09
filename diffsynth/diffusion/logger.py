import os, json, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, config=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.config = config
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            self._save_config(self.output_path)


    def _save_config(self, output_dir):
        if self.config is None:
            return
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=True)


    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps=None,
        loss=None,
        grad_norm=None,
        optimizer=None,
        epoch=None,
        force_step=True,
    ):
        if not force_step:
            return

        self.num_steps += 1

        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

        if self.num_steps % 1 == 0:
            metrics = {}
            if loss is not None:
                metrics["train_loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if optimizer is not None:
                metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            if epoch is not None:
                metrics["epoch"] = epoch

            if len(metrics) > 0:
                accelerator.log(metrics, step=self.num_steps)
                metric_str = " ".join(f"{k}={v}" for k, v in metrics.items())
                accelerator.print(f"[step {self.num_steps}] {metric_str}")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        checkpoint_dir = os.path.join(self.output_path, f"epoch-{epoch_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if accelerator.is_main_process:
            self._save_config(checkpoint_dir)
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            path = os.path.join(checkpoint_dir, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            self._save_config(self.output_path)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
