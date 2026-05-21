# Development Environment

## Platform split

- **Development machine**: macOS (where this code is written and Claude Code runs)
- **Target machine**: Linux workstation (where the code actually executes)

## Testing policy

Unless explicitly told "we are on the workstation right now", assume we are on the Mac and do not attempt to run, test, or validate anything that requires the workstation environment (e.g. MuJoCo simulation, GPU training, Linux-only tools). Acknowledge when something cannot be verified locally rather than skipping or faking it.