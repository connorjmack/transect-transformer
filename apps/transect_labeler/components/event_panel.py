"""Event panel component for displaying event info at current location."""

import streamlit as st

from apps.transect_labeler import config
from apps.transect_labeler.utils.event_loader import mop_to_transect_id


def render_event_panel():
    """Render the event information panel."""
    if st.session_state.data is None:
        return

    st.subheader("Event Info")

    # Check if we have event data
    events_df = st.session_state.get('events_df')
    if events_df is None or events_df.empty:
        st.info("No events loaded")
        return

    # Get current event info
    current_events = st.session_state.get('current_events', [])

    # Show current transect MOP info
    from apps.transect_labeler.utils.data_loader import get_transect_ids
    transect_ids = get_transect_ids(st.session_state.data)
    current_tid = transect_ids[st.session_state.current_transect_idx]
    st.text(f"Transect: {current_tid}")

    if current_events:
        st.success(f"**{len(current_events)} event(s) here**")

        for i, event in enumerate(current_events):
            with st.expander(f"Event {i+1}: {event['volume']:.1f} m³", expanded=(i == 0)):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Volume", f"{event['volume']:.1f} m³")
                    st.metric("Width", f"{event['width']:.1f} m")
                with col2:
                    st.metric("Elevation", f"{event['elevation']:.1f} m")
                    st.metric("Height", f"{event['height']:.1f} m")

                st.text(f"Beach: {event['beach'].title()}")
                st.text(f"Dates: {event['start_date']} to {event['end_date']}")

                # Show which MOP the event is centered on
                expected_tid = mop_to_transect_id(event['mop_centroid'])
                st.text(f"Event MOP: {expected_tid}")
    else:
        # Check if we're on an event in the queue
        event_queue = st.session_state.get('event_queue', [])
        current_transect = st.session_state.current_transect_idx
        current_pair = st.session_state.current_pair_idx

        in_queue = False
        for item in event_queue:
            if item['transect_idx'] == current_transect and item['pair_idx'] == current_pair:
                in_queue = True
                event = item['event']
                st.warning("**Event from queue** (dates may not match exactly)")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Volume", f"{event['volume']:.1f} m³")
                    st.metric("Width", f"{event['width']:.1f} m")
                with col2:
                    st.metric("Elevation", f"{event['elevation']:.1f} m")
                    st.metric("Height", f"{event['height']:.1f} m")
                st.text(f"Beach: {event['beach'].title()}")
                st.text(f"Dates: {event['start_date']} to {event['end_date']}")
                break

        if not in_queue:
            st.info("No known events at this transect-pair")

    st.markdown("---")

    # Show queue position
    _render_queue_position()


def _render_queue_position():
    """Show current position in event queue."""
    event_queue = st.session_state.get('event_queue', [])

    if not event_queue:
        st.text("No event queue")
        return

    # Find current position in queue
    current_transect = st.session_state.current_transect_idx
    current_pair = st.session_state.current_pair_idx

    current_queue_idx = None
    for i, item in enumerate(event_queue):
        if item['transect_idx'] == current_transect and item['pair_idx'] == current_pair:
            current_queue_idx = i
            break

    total_events = len(event_queue)
    labeled_events = 0
    if st.session_state.labels is not None:
        for item in event_queue:
            if st.session_state.labels[item['transect_idx'], item['pair_idx']] != config.UNLABELED_VALUE:
                labeled_events += 1

    if current_queue_idx is not None:
        st.text(f"Queue: #{current_queue_idx + 1} of {total_events}")
    else:
        st.text(f"Queue: Not on event ({total_events} total)")

    st.text(f"Labeled: {labeled_events}/{total_events}")
