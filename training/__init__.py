"""
TKS-LLM Training Module

Components:
    - losses: TKSLoss and individual loss components
    - trainer: TKSTrainer and StagedTrainer classes
    - datasets: TKS dataset classes for training
    - curriculum: CurriculumLossScheduler for staged training
"""

from .losses import (
    TKSLoss,
    TKSLossConfig,
    TaskLoss,
    RPMLoss,
    AttractorLoss,
    InvolutionLoss,
    SpectralLoss,
    CascadeLoss,
    CurriculumLossScheduler,
    INVOLUTION_PAIRS,
)

from .trainer import (
    TrainingConfig,
    TrainingState,
    EvaluationResult,
    TKSTrainer,
    StagedTrainer,
    create_optimizer,
    create_scheduler,
    load_checkpoint,
    get_device,
)

from .datasets import (
    TKSBaseDataset,
    TKSElementsDataset,
    TKSCompositionsDataset,
    TKSRPMDataset,
    TKSMultiTaskDataset,
    PilotDataset,
    tks_collate_fn,
    generate_pilot_data,
    element_to_index,
    index_to_element,
    compute_rpm_from_noetics,
    WORLD_OFFSETS,
    NOETIC_NAMES,
)

__all__ = [
    # Losses
    'TKSLoss',
    'TKSLossConfig',
    'TaskLoss',
    'RPMLoss',
    'AttractorLoss',
    'InvolutionLoss',
    'SpectralLoss',
    'CascadeLoss',
    'CurriculumLossScheduler',
    'INVOLUTION_PAIRS',
    # Trainer
    'TrainingConfig',
    'TrainingState',
    'EvaluationResult',
    'TKSTrainer',
    'StagedTrainer',
    'create_optimizer',
    'create_scheduler',
    'load_checkpoint',
    'get_device',
    # Datasets
    'TKSBaseDataset',
    'TKSElementsDataset',
    'TKSCompositionsDataset',
    'TKSRPMDataset',
    'TKSMultiTaskDataset',
    'PilotDataset',
    'tks_collate_fn',
    'generate_pilot_data',
    'element_to_index',
    'index_to_element',
    'compute_rpm_from_noetics',
    'WORLD_OFFSETS',
    'NOETIC_NAMES',
]
