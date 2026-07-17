# Carbon capture paper

## Data storage and tracking

This repository uses [DVC](https://dvc.org/) to keep large and generated data
out of Git while still making the project reproducible. Git tracks the DVC
metadata files, and DVC stores the actual tracked data in the configured remote.
Contributors should install the repository's pre-commit hooks so DVC can handle
common synchronization steps automatically.

All project data lives under `data/`:

- `data/output/` contains generated artifacts derived from external data sources.
  This directory is tracked by DVC and shared through the remote so contributors
  can reproduce or inspect generated outputs without committing large files to
  Git.


### One-time setup

1. Clone the repository and install the Python environment as usual:

   ```sh
   uv sync
   ```

2. Install the repository hooks:

   ```sh
   uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type post-checkout
   ```

   The DVC hooks are already declared in `.pre-commit-config.yaml`. Installing
   them enables:

   - `pre-commit`: runs `dvc status` before `git commit`, so DVC changes are
     visible before metadata is committed.
   - `pre-push`: runs `dvc push` before `git push`, so updated generated data is
     uploaded before the Git branch is published.
   - `post-checkout`: runs `dvc checkout` after `git checkout`, so the local
     workspace is updated to match the DVC metadata on the checked-out branch
     when the data is already in the local DVC cache.

3. Setup the connection string for the DVC remote:

   ```sh
   uv run dvc remote modify azure_remote --local connection-string <your-azure-connection-string>
   ```

This connection string is stored in `.dvc/config.local` and is not shared with Git. It should be set once per machine.


VS Code users may also want the
[DVC by lakeFS](https://marketplace.visualstudio.com/items?itemName=lakefs.lakefs-dvc)
extension. It adds DVC status, tracked-data views, and sync actions to the IDE.

### Getting data after cloning

After cloning the repository for the first time, pull the DVC-tracked generated
data:

```sh
uv run dvc pull
```

This restores tracked generated artifacts under `data/output`. 

After switching branches, the installed `post-checkout` hook runs `dvc checkout`
to align the workspace with the checked-out DVC metadata. If the required
generated data is not yet in your local DVC cache, run `dvc pull`.

### Updating generated data

When your work changes generated artifacts:

1. Update or regenerate files under `data/output/`.
2. Record the new DVC state:

   ```sh
   uv run dvc add data/output/<path-to-generated-file>
   ```

3. Review what changed:

   ```sh
   git status
   dvc status
   ```

4. Commit the DVC metadata, not the generated files themselves:

   ```sh
   git add data/output/<path-to-generated-file>.dvc
   git commit -m "Update generated data"
   ```

   The installed `pre-commit` hook runs `dvc status` during `git commit`. If it
   reports that generated data and DVC metadata are out of sync, update the DVC
   metadata with `dvc add data` before committing.

5. Push your branch as usual. The installed `pre-push` hook runs `dvc push`
   before Git publishes the branch:

   ```sh
   git push
   ```

### DVC workflow tips

- Use `dvc status` to compare local data with the current DVC metadata.
- Use `dvc pull` when a branch references generated data that is missing from
  your local cache.
- Let the `pre-push` hook run `dvc push` when a commit updates `data.dvc`;
  otherwise collaborators may receive metadata for data that is not yet
  available in the remote.
- Do not manually commit files from `data/` to Git. Git should track DVC
  metadata, notebooks, scripts, and documentation; DVC should track generated
  data artifacts.
