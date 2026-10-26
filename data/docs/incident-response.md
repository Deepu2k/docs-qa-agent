# Incident Response

An incident is any unplanned event that degrades a customer-facing service, puts data at risk, or could cause a robot to behave unsafely. This document explains how we declare, run and learn from incidents.

## Severity levels

- SEV1: A safety risk, a data breach, or a full outage of the fleet control service. Response starts immediately, day or night, and the incident commander updates the status page every 30 minutes.
- SEV2: A major feature is broken for many customers, or a single customer's fleet is down. Response starts within 30 minutes during business hours and within 1 hour outside them.
- SEV3: A minor bug or a degraded feature with a workaround. Handled during business hours by the owning team.

## Declaring an incident

Anyone can declare an incident. Type /incident in the chat tool and choose a severity. If you are unsure of the severity, choose the higher one; it is easy to downgrade later. Declaring an incident creates a channel, pages the on-call engineer, and starts a timeline.

## Roles

Every SEV1 and SEV2 incident has an incident commander, who coordinates the response and makes decisions, and a communications lead, who updates customers and the status page. The incident commander should not be the person debugging the problem.

## Customer communication

For SEV1 incidents, customers are notified within 1 hour. Updates continue every 30 minutes until the incident is resolved. Support handles individual customer questions using the approved status page wording.

## Postmortems

A written postmortem is required for every SEV1 and SEV2 incident, within 5 business days of resolution. Postmortems are blameless: they focus on how systems and processes failed, not on who made a mistake. Each postmortem lists action items with owners and due dates, and these are tracked in the weekly reliability review.

## Safety incidents

If a robot behaves unsafely at a customer site, the field team triggers the remote stop for the affected fleet first and declares a SEV1 second. Safety always comes before diagnosis.
