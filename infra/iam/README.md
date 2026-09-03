# IAM policies the deploy workflow depends on

None of the AWS Batch infrastructure is managed as code. These files are a recorded copy
of the policy documents the deploy workflow needs, so that a lost or rebuilt account can
be restored from something reviewable rather than from prose. They are not applied
automatically: nothing here runs, and editing a file changes nothing until someone applies
it deliberately.

| File | Attached to | Purpose |
| --- | --- | --- |
| `github-action-role.SrmAwsBatchDeployPolicy.json` | inline policy on `github-action-role` | what CI needs to submit, poll, tag and terminate Batch jobs, register job definitions, read ECR and read job logs |
| `coiled-carbonplan.trust-policy.json` | trust policy on `coiled-carbonplan` | admits `ecs-tasks.amazonaws.com` so Batch can assume it as a job role, alongside `ec2.amazonaws.com`, which Coiled clusters still use |
| `srm-batch-job-role.SrmS3Access.json` | inline policy on `srm-batch-job-role` | the tighter task role, kept for reference; production runs as `coiled-carbonplan` because source.coop grants write to that principal by name |

## Applying a change

```bash
aws iam put-role-policy --role-name github-action-role \
  --policy-name SrmAwsBatchDeployPolicy \
  --policy-document file://infra/iam/github-action-role.SrmAwsBatchDeployPolicy.json

aws iam update-assume-role-policy --role-name coiled-carbonplan \
  --policy-document file://infra/iam/coiled-carbonplan.trust-policy.json
```

## Keeping them honest

A file here can drift from the account without anything failing, which is the main risk of
recording rather than managing. There is no automated check: verifying these in-workflow
would need `iam:SimulatePrincipalPolicy` on `github-action-role`, which it does not have,
and granting it would be a wider permission than any it verifies. Run the
`simulate-principal-policy` commands in deploy.md by hand after changing the role. Refresh
a file with
`aws iam get-role-policy ... --query PolicyDocument`, and see
[deploy.md](../../docs/how-to/deploy.md) for what each statement is for.
