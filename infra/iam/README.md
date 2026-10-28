# IAM policies the deploy workflow depends on

None of the AWS Batch infrastructure is managed as code. These files are a recorded copy
of the policy documents the deploy workflow needs, so that a lost or rebuilt account can
be restored from something reviewable rather than from prose. They are not applied
automatically: nothing here runs, and editing a file changes nothing until someone applies
it deliberately.

| File | Attached to | Purpose |
| --- | --- | --- |
| `gha-sai-downscaling-deploy.trust-policy.json` | trust policy on `gha-sai-downscaling-deploy` | admits this repository's OIDC subjects for any branch, tag or environment, not pull requests |
| `gha-sai-downscaling-deploy.policy.json` | inline policy `SrmDeploy` on `gha-sai-downscaling-deploy` | what deploy, build and processing jobs need: submit, poll, tag and terminate Batch jobs, register job definitions, push and read the ECR image, read job logs, read and write `carbonplan-srm`, read input buckets |
| `gha-sai-downscaling-ci.trust-policy.json` | trust policy on `gha-sai-downscaling-ci` | admits this repository's pull requests and branches |
| `gha-sai-downscaling-ci.policy.json` | inline policy `SrmCiRead` on `gha-sai-downscaling-ci` | read-only access to `carbonplan-srm` and the input buckets for tests |
| `coiled-carbonplan.trust-policy.json` | trust policy on `coiled-carbonplan` | admits `ecs-tasks.amazonaws.com` so Batch can assume it as a job role, alongside `ec2.amazonaws.com`, which Coiled clusters still use |
| `srm-batch-job-role.SrmS3Access.json` | inline policy on `srm-batch-job-role` | the tighter task role, kept for reference; production runs as `coiled-carbonplan` because source.coop grants write to that principal by name |

## Applying a change

```bash
aws iam update-assume-role-policy --role-name gha-sai-downscaling-deploy \
  --policy-document file://infra/iam/gha-sai-downscaling-deploy.trust-policy.json
aws iam put-role-policy --role-name gha-sai-downscaling-deploy \
  --policy-name SrmDeploy \
  --policy-document file://infra/iam/gha-sai-downscaling-deploy.policy.json

aws iam update-assume-role-policy --role-name gha-sai-downscaling-ci \
  --policy-document file://infra/iam/gha-sai-downscaling-ci.trust-policy.json
aws iam put-role-policy --role-name gha-sai-downscaling-ci \
  --policy-name SrmCiRead \
  --policy-document file://infra/iam/gha-sai-downscaling-ci.policy.json

aws iam update-assume-role-policy --role-name coiled-carbonplan \
  --policy-document file://infra/iam/coiled-carbonplan.trust-policy.json
```

## Keeping them honest

A file here can drift from the account without anything failing, which is the main risk of
recording rather than managing. There is no automated check: verifying these in-workflow
would need `iam:SimulatePrincipalPolicy` on `gha-sai-downscaling-deploy`, which it does not have,
and granting it would be a wider permission than any it verifies. Run the
`simulate-principal-policy` commands in deploy.md by hand after changing the role. Refresh
a file with
`aws iam get-role-policy ... --query PolicyDocument`, and see
[deploy.md](../../docs/how-to/deploy.md) for what each statement is for.
