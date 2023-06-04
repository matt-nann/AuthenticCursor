while IFS= read -r line || [[ -n "$line" ]]; do
    heroku config:set "$line" --app seek-spira
done < "cloud.env"