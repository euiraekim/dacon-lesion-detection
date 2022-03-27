checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', interval=200)
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
