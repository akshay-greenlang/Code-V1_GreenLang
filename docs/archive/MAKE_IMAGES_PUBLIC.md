# Make Docker Images Public

The images have been successfully built and pushed to GHCR, but they're currently **private**.

## To Make Images Public (Required for DoD):

1. **Go to your packages**: https://github.com/akshay-greenlang?tab=packages

2. **Click on** `greenlang-runner`

3. **Click** "Package settings" (right side, gear icon)

4. **Scroll down to** "Danger Zone"

5. **Click** "Change visibility"

6. **Select** "Public"

7. **Type** the package name to confirm: `greenlang-runner`

8. **Click** "I understand the consequences, change package visibility"

9. **Repeat for** `greenlang-full` package (if it exists)

## After Making Public:

The images will be available at:
- `ghcr.io/akshay-greenlang/greenlang-runner:0.2.0`
- `ghcr.io/akshay-greenlang/greenlang-runner:latest`
- `ghcr.io/akshay-greenlang/greenlang-full:0.2.0`
- `ghcr.io/akshay-greenlang/greenlang-full:latest`

## Test Pull Command:
```bash
docker pull ghcr.io/akshay-greenlang/greenlang-runner:0.2.0
docker run --rm ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 --version
```

## Verify Multi-arch:
```bash
docker buildx imagetools inspect ghcr.io/akshay-greenlang/greenlang-runner:0.2.0
```

This should show both `linux/amd64` and `linux/arm64` platforms.