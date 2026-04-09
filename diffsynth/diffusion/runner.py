import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    max_grad_norm: float = 1.0,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_grad_norm = args.max_grad_norm
        resume_from = getattr(args, "resume_from", None)
        ckpt_path = getattr(args, "ckpt_path", None)
        dataset_repeat = getattr(args, "dataset_repeat", 1)
    if resume_from and ckpt_path:
        raise ValueError("`--resume_from` and `--ckpt_path` are mutually exclusive. Use only one of them.")
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if args is not None and len(getattr(accelerator, "log_with", [])) > 0:
        tracker_init_kwargs = {}
        if getattr(args, "use_wandb", False):
            tracker_init_kwargs["wandb"] = {"name": args.output_path}
        if getattr(args, "use_swanlab", False):
            experiment_name = getattr(args, "swanlab_experiment_name", None) or args.output_path
            tracker_init_kwargs["swanlab"] = {"experiment_name": experiment_name}
        if len(tracker_init_kwargs) > 0:
            accelerator.init_trackers(
                project_name="DiffSynth-Studio",
                config=vars(args),
                init_kwargs=tracker_init_kwargs,
            )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    start_epoch = 0
    if resume_from:
        accelerator.print(f"Resuming training from checkpoint: {resume_from}")
        accelerator.load_state(resume_from)
        basename = os.path.basename(os.path.normpath(resume_from))
        if basename.startswith("epoch-"):
            try:
                resume_epoch_label = int(basename.split("-", 1)[1])
                if dataset_repeat > 1 and (resume_epoch_label + 1) % dataset_repeat == 0:
                    start_epoch = (resume_epoch_label + 1) // dataset_repeat
                else:
                    start_epoch = resume_epoch_label + 1
            except ValueError:
                start_epoch = 0
        state_step = getattr(accelerator.state, "step", None)
        if state_step is not None:
            model_logger.num_steps = state_step
    
    for epoch_id in range(start_epoch, num_epochs):
        train_loss = 0.0
        loss_count = 0
        optimizer.zero_grad()
        epoch_label = (epoch_id + 1) * dataset_repeat - 1

        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)

                train_loss += loss.detach().float().item()
                loss_count += 1

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients and max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm = grad_norm.item()

                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    avg_loss = train_loss / loss_count if loss_count > 0 else None
                    model_logger.on_step_end(
                        accelerator,
                        model,
                        save_steps,
                        loss=avg_loss,
                        grad_norm=grad_norm,
                        optimizer=optimizer,
                        epoch=epoch_label,
                        force_step=True,
                    )

                    train_loss = 0.0
                    loss_count = 0
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_label)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 32,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
