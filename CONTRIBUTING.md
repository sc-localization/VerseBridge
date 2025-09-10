> [!NOTE]
> Please prefer English language for all communication.

## Creating an issue

Before creating an issue please ensure that the problem is not [already reported](https://github.com/sc-localization/VerseBridge/issues).


## How to Contribute

Thank you for your interest in contributing to the Star Citizen Translator project! To help you get started, please follow these steps:

1. **Fork and Clone the Repository**

   First, create your own copy of the repository by clicking the "Fork" button on GitHub. Then, clone your fork to your local machine:

   ```sh
   git clone https://github.com/sc-localization/VerseBridge.git
   cd VerseBridge
   ```

2. **Create a New Branch**

   Create a new branch for your feature or bugfix. Use a descriptive name for your branch:

   ```sh
   git checkout -b your-feature-name
   ```

3. **Install Dependencies**

   Install the required Python dependencies using [uv](https://github.com/astral-sh/uv). Make sure you have Python and uv installed:

   ```sh
   uv sync
   ```

4. **Make Your Changes**

   Implement your feature or fix the bug. Be sure to follow the project's coding style and add tests if necessary.

5. **Commit Your Changes**

   Stage and commit your changes with a clear and descriptive commit message:

   ```sh
   git add .
   git commit -m "Describe your changes here"
   ```

6. **Push to Your Fork**

   Push your branch to your forked repository on GitHub:

   ```sh
   git push origin your-feature-name
   ```

7. **Open a Pull Request**

   Go to the original repository and open a Pull Request (PR) from your branch. Include a detailed description of your changes and reference any related issues.

## Commit messages

Commit messages should follow the [Conventional Commits](https://conventionalcommits.org) specification:

```
<type>[optional scope]: <description>
```

### Allowed `<type>`

- `chore`: any repository maintainance changes
- `feat`: code change that adds a new feature
- `fix`: bug fix
- `perf`: code change that improves performance
- `refactor`: code change that is neither a feature addition nor a bug fix nor a performance improvement
- `docs`: documentation only changes
- `ci`: a change made to CI configurations and scripts
- `style`: cosmetic code change
- `test`: change that only adds or corrects tests
- `revert`: change that reverts previous commits

If you have any questions or need help, feel free to open an issue or ask in the discussions section. We appreciate your contributions!
