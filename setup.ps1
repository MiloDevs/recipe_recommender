Write-Host "Setting up the environment..."

# Create virtual environment
python -m venv .venv

# Activate the environment
try {
    .\.venv\Scripts\Activate.ps1
}
catch {
    Write-Host "Failed to activate virtual environment. Please activate it manually."
    exit 1
}

Write-Host "Virtual environment activated."

# Install requirements
Write-Host "Installing the requirements..."
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install requirements. Please check your requirements.txt file."
    exit 1
}

Write-Host "Setup complete!"
Write-Host "My work is complete, I shall now leave, farewell!"