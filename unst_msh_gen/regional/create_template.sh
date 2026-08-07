#!/bin/bash

# define template and config
CONFIG_FILE="config_gjs_reg.env"
TEMPLATE_FILE="config_gjs_reg.template"
OUTPUT_FILE="config_gjs_reg.ini"

# Verify that required files exist before continuing
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found." >&2
    exit 1
fi

if [[ ! -f "$TEMPLATE_FILE" ]]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found." >&2
    exit 1
fi

# Copy the original template to the new output file destination
cp "$TEMPLATE_FILE" "$OUTPUT_FILE"

# Read the config file line by line
while IFS='=' read -r key value || [[ -n "$key" ]]; do
    # Strip any leading/trailing whitespace from the key
    key=$(echo "$key" | xargs)

    # Skip lines that are empty or start with a comment (#)
    if [[ -z "$key" || "$key" =~ ^# ]]; then
        continue
    fi

    # Strip leading/trailing whitespace and optional quotes from the value
    value=$(echo "$value" | xargs | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

    # Safely escape special characters for the sed substitution command
    # This prevents errors if your values contain slashes or backslashes
    escaped_value=$(printf '%s\n' "$value" | sed 's/[&/\]/\\&/g')

    # Replace the placeholder {{KEY}} with the configuration value in-place
    # (Modifies the output file copy, keeping the original template untouched)
    sed -i "s/{{$key}}/$escaped_value/g" "$OUTPUT_FILE"

done < "$CONFIG_FILE"

echo "Success! Generated file saved to: $OUTPUT_FILE"
