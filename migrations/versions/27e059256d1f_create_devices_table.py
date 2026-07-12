"""create devices table

Revision ID: 27e059256d1f
Revises: 76f0ca6c6e6b
Create Date: 2026-07-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '27e059256d1f'
down_revision: Union[str, Sequence[str], None] = '76f0ca6c6e6b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('devices',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('platform', sa.String(length=10), nullable=False),
    sa.Column('manufacturer', sa.String(length=100), nullable=True),
    sa.Column('device_model', sa.String(length=100), nullable=True),
    sa.Column('os_version', sa.String(length=50), nullable=True),
    sa.Column('call_recording_supported', sa.Boolean(), nullable=False),
    sa.Column('on_device_model_version', sa.String(length=50), nullable=True),
    sa.Column('embedding_synced_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('push_token', sa.String(length=255), nullable=True),
    sa.Column('notification_enabled', sa.Boolean(), nullable=False),
    sa.Column('notification_permission', sa.Boolean(), nullable=False),
    sa.Column('microphone_permission', sa.Boolean(), nullable=False),
    sa.Column('file_permission', sa.Boolean(), nullable=False),
    sa.Column('battery_optimization_ignored', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("platform IN ('android', 'ios')", name='ck_devices_platform'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_devices_user_id'), 'devices', ['user_id'], unique=False)
    op.create_foreign_key(
        'fk_refresh_tokens_device_id_devices',
        'refresh_tokens', 'devices',
        ['device_id'], ['id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('fk_refresh_tokens_device_id_devices', 'refresh_tokens', type_='foreignkey')
    op.drop_index(op.f('ix_devices_user_id'), table_name='devices')
    op.drop_table('devices')
