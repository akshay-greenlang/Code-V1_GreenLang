# 🚀 Complete Beginner's Guide to Upload GreenLang to PyPI

## 📍 Step 1: Open the Right Folder

### Option A: Using File Explorer (EASIEST)
1. Open File Explorer
2. Copy this path: `C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang`
3. Paste it in the address bar and press Enter
4. You should see files like `pyproject.toml`, `README.md`, and a folder called `dist`
5. **Double-click** on `UPLOAD_TO_PYPI_SIMPLE.bat`
6. Skip to Step 3 below

### Option B: Using Command Prompt
1. Press `Windows Key + R`
2. Type `cmd` and press Enter
3. Copy and paste this command:
```cmd
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
```
4. Press Enter
5. Continue to Step 2

## 📍 Step 2: Run the Upload Command

Copy and paste this EXACT command:
```cmd
python -m twine upload dist\greenlang-0.2.0-py3-none-any.whl dist\greenlang-0.2.0.tar.gz
```
Press Enter

## 📍 Step 3: Enter Your Credentials

### When you see "Enter your username:"
1. Type exactly: `__token__`
2. Press Enter

### When you see "Enter your password:"
1. Paste your PyPI token (Ctrl+V or right-click → Paste)
2. **IMPORTANT**: The token won't show on screen - this is normal!
3. Just paste it and press Enter even though you can't see it

## 📍 Step 4: Wait for Upload

You'll see progress bars:
```
Uploading greenlang-0.2.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## ✅ Success!

If successful, you'll see:
```
View at: https://pypi.org/project/greenlang/0.2.0/
```

---

# 🆘 If Something Goes Wrong

## Error: "Cannot find file"
You're in the wrong folder. Do this:
1. Close the command prompt
2. Start over from Step 1
3. Make sure you're in the folder with the `dist` folder

## Error: "403 Forbidden"
Your token might be wrong. Check:
1. Token starts with `pypi-`
2. You typed username as `__token__` (not your actual username)
3. Token hasn't expired

## Error: "File already exists"
Version 0.2.0 might already be uploaded. Check:
https://pypi.org/project/greenlang/

---

# 🎯 Quick Copy-Paste Commands

Just copy and run these one by one:

```cmd
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
```

Then:
```cmd
python -m twine upload dist\greenlang-0.2.0-py3-none-any.whl dist\greenlang-0.2.0.tar.gz
```

When prompted:
- Username: `__token__`
- Password: [Your token - paste it even though it's invisible]

---

# 🎬 Complete Example

Here's exactly what your screen should look like:

```
C:\Users\rshar>cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"

C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang>python -m twine upload dist\greenlang-0.2.0-py3-none-any.whl dist\greenlang-0.2.0.tar.gz
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: __token__
Enter your password:
Uploading greenlang-0.2.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 549.6/549.6 kB
Uploading greenlang-0.2.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 580.1/580.1 kB

View at:
https://pypi.org/project/greenlang/0.2.0/
```

That's it! Your package is now on PyPI! 🎉